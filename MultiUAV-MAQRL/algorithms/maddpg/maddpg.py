"""Vanilla centralized-training/decentralized-execution MADDPG learner."""
from dataclasses import asdict
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from config.maddpg_config import MADDPGConfig
from .networks import MADDPGNetworks, soft_update
from .replay_buffer import ReplayBuffer


def bellman_target(rewards, next_q, terminated, gamma):
    """Time limits bootstrap from final states; true terminations never do."""
    return rewards+gamma*(~terminated.bool()).to(next_q.dtype)*next_q


def resolve_device(request):
    if request=='auto':
        request='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if request=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if request=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    return torch.device(request)


class MADDPG:
    def __init__(self, agent_ids, global_state_dim, config=None, observation_dim=157):
        self.config=copy.deepcopy(config or MADDPGConfig());self.config.validate()
        self.agent_ids=tuple(agent_ids)
        self.device=resolve_device(self.config.device)
        self.global_state_dim,self.observation_dim=global_state_dim,observation_dim
        # Seed initialization without changing the caller's CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.networks=MADDPGNetworks(len(self.agent_ids),global_state_dim,self.config,observation_dim).to(self.device)
        self.actor_optimizers=[torch.optim.Adam(m.parameters(),lr=self.config.actor_lr) for m in self.networks.actors]
        self.critic_optimizers=[torch.optim.Adam(m.parameters(),lr=self.config.critic_lr) for m in self.networks.critics]
        self.replay=ReplayBuffer(self.config.replay_capacity,self.agent_ids,observation_dim,global_state_dim,self.config.seed)
        self.rng=np.random.default_rng(self.config.seed)
        self.environment_steps=0;self.update_steps=0
        self.noise_std=self.config.exploration_noise_std

    def select_actions(self, observations, training=False):
        if not set(observations).issubset(self.agent_ids):
            raise ValueError('Unknown actor ID')
        actions={}
        with torch.no_grad():
            for i,agent in enumerate(self.agent_ids):
                if agent not in observations: continue
                value=np.asarray(observations[agent],dtype=np.float32)
                if value.shape!=(self.observation_dim,) or not np.isfinite(value).all():
                    raise ValueError('Invalid local observation')
                action=self.networks.actors[i](torch.as_tensor(value,device=self.device)).cpu().numpy()
                if training: action=action+self.rng.normal(0,self.noise_std,3)
                actions[agent]=np.clip(action,-1,1).astype(np.float32)
        return actions

    def record_transition(self,*args,**kwargs):
        """Forward EXACT executed actions/final next states to joint replay."""
        self.replay.add(*args,**kwargs)
        self.environment_steps+=1
        if self.environment_steps>self.config.warmup_steps:
            self.noise_std=max(self.config.minimum_exploration_noise,self.noise_std*self.config.exploration_noise_decay)

    @torch.no_grad()
    def targets(self,batch):
        actions=[]
        for i,actor in enumerate(self.networks.target_actors):
            # Truncated agents retain final-state target actions, terminal ones do not.
            valid=batch['next_valid_agents'][:,i] & ~batch['terminated'][:,i]
            actions.append(actor(batch['next_observations'][:,i])*valid[:,None])
        joint=torch.cat(actions,dim=-1)
        return torch.stack([bellman_target(batch['rewards'][:,i],
            critic(batch['next_global_state'],joint).squeeze(-1),batch['terminated'][:,i],self.config.gamma)
            for i,critic in enumerate(self.networks.target_critics)],dim=1)

    def actor_loss(self,batch,index):
        """Only actor index has a gradient path; other actions come from replay."""
        actions=[batch['actions'][:,j].detach() if j!=index else
                 self.networks.actors[index](batch['observations'][:,index])
                 for j in range(len(self.agent_ids))]
        q=self.networks.critics[index](batch['global_state'],torch.cat(actions,dim=-1)).squeeze(-1)
        mask=batch['valid_agents'][:,index].to(q.dtype)
        return -(q*mask).sum()/mask.sum().clamp_min(1)

    def _step(self,loss,optimizer,module):
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(module.parameters(),self.config.gradient_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def update(self,batch=None):
        if batch is None:
            if (self.environment_steps<self.config.warmup_steps or
                self.environment_steps%self.config.policy_update_frequency or len(self.replay)<self.config.batch_size):
                return None
            batch=self.replay.sample(self.config.batch_size,self.device)
        batch={k:v.to(self.device) for k,v in batch.items()}
        target=self.targets(batch)
        metrics={}
        for i,agent in enumerate(self.agent_ids):
            mask=batch['valid_agents'][:,i].float()
            if not mask.any(): continue
            critic=self.networks.critics[i]
            q=critic(batch['global_state'],batch['actions'].flatten(1)).squeeze(-1)
            critic_loss=((q-target[:,i]).square()*mask).sum()/mask.sum()
            self._step(critic_loss,self.critic_optimizers[i],critic)
            critic.requires_grad_(False)
            try:
                actor_loss=self.actor_loss(batch,i)
                self._step(actor_loss,self.actor_optimizers[i],self.networks.actors[i])
            finally:
                critic.requires_grad_(True)
            metrics[agent]=dict(actor_loss=float(actor_loss.detach()),critic_loss=float(critic_loss.detach()))
            soft_update(self.networks.target_actors[i],self.networks.actors[i],self.config.tau)
            soft_update(self.networks.target_critics[i],critic,self.config.tau)
        self.update_steps+=1
        return metrics

    def save_checkpoint(self,path,training_state=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(training_state=training_state,version=1,config=asdict(self.config),agent_ids=self.agent_ids,
            global_state_dim=self.global_state_dim,observation_dim=self.observation_dim,
            networks=self.networks.state_dict(),actors=[o.state_dict() for o in self.actor_optimizers],
            critics=[o.state_dict() for o in self.critic_optimizers],environment_steps=self.environment_steps,
            update_steps=self.update_steps,noise_std=self.noise_std,rng=self.rng.bit_generator.state,
            replay=self.replay._data,replay_cursor=self.replay._cursor,replay_rng=self.replay.rng.bit_generator.state),path)

    @classmethod
    def load_checkpoint(cls,path,device=None):
        """Load only trusted local checkpoints (torch pickle includes replay arrays)."""
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['version']!=1: raise ValueError('Unsupported checkpoint version')
        config=MADDPGConfig(**data['config'])
        if device is not None: config.device=device
        model=cls(data['agent_ids'],data['global_state_dim'],config,data['observation_dim'])
        model.networks.load_state_dict(data['networks'])
        for optimizers,key in [(model.actor_optimizers,'actors'),(model.critic_optimizers,'critics')]:
            for optimizer,state in zip(optimizers,data[key]): optimizer.load_state_dict(state)
        model.environment_steps=data['environment_steps'];model.update_steps=data['update_steps']
        model.noise_std=data['noise_std'];model.rng.bit_generator.state=data['rng']
        model.replay._data=data['replay'];model.replay._cursor=data['replay_cursor']
        model.replay.rng.bit_generator.state=data['replay_rng']
        model.training_state=data.get("training_state")
        return model
