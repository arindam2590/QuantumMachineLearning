"""Vanilla agent-specific MADDPG with joint uniform replay and PyTorch Q critics."""
import copy,random
from dataclasses import asdict
from collections import deque
import numpy as np
import torch
from torch import nn
from config.maddpg_config import MADDPGConfig

def mlp(inputs,outputs,hidden,tanh=False):
    layers=[]
    for width in hidden:layers += [nn.Linear(inputs,width),nn.ReLU()];inputs=width
    layers.append(nn.Linear(inputs,outputs))
    if tanh:layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class MADDPG:
    def __init__(self,ids,config=None):
        self.config=config or MADDPGConfig();self.config.validate();self.ids=tuple(ids)
        self.state_dim=158*len(ids);self.steps=0;self.updates=0
        self.rng=np.random.default_rng(self.config.seed)
        with torch.random.fork_rng():
            torch.manual_seed(self.config.seed)
            self.actors=nn.ModuleList([mlp(157,3,self.config.hidden_layers,True) for _ in ids])
            self.critics=nn.ModuleList([mlp(self.state_dim+3*len(ids),1,self.config.hidden_layers) for _ in ids])
        self.target_actors=copy.deepcopy(self.actors);self.target_critics=copy.deepcopy(self.critics)
        self.actor_opts=[torch.optim.Adam(a.parameters(),lr=self.config.actor_lr) for a in self.actors]
        self.critic_opts=[torch.optim.Adam(c.parameters(),lr=self.config.critic_lr) for c in self.critics]
        self.replay=deque(maxlen=self.config.replay_capacity)

    def actions(self,obs,training=False):
        with torch.no_grad():
            return {a:np.clip(self.actors[i](torch.as_tensor(obs[a],dtype=torch.float32)).numpy()+
                 (self.rng.normal(0,self.config.noise_std,3) if training else 0),-1,1).astype(np.float32)
                 for i,a in enumerate(self.ids) if a in obs}

    def add(self,obs,state,actions,rewards,next_obs,next_state,term,trunc,active):
        n=len(self.ids)
        row=(obs.copy(),state.copy(),np.stack([actions.get(a,np.zeros(3)) for a in self.ids]).astype(np.float32),
            np.array([rewards.get(a,0) for a in self.ids],np.float32),next_obs.copy(),next_state.copy(),
            np.array([term.get(a,False) for a in self.ids]),np.array([trunc.get(a,False) for a in self.ids]),
            np.array([a in active for a in self.ids],np.float32))
        self.replay.append(row);self.steps+=1

    def update(self):
        if not getattr(self,'replay_compatible',True):
            raise ValueError('Historical replay uses an older critic encoding; inference only')
        c=self.config
        if self.steps<c.warmup_steps or self.steps%c.update_interval or len(self.replay)<c.batch_size:return None
        indices=self.rng.choice(len(self.replay),c.batch_size,replace=False)
        batch=[self.replay[int(i)] for i in indices]
        o,s,a,r,no,ns,t,tr,mask=[torch.as_tensor(np.stack(x),dtype=torch.float32) for x in zip(*batch)]
        # A truncation bootstraps from its final observation; retired terminals do not.
        next_mask=torch.maximum(ns[:,-len(self.ids):],tr)*(1-t)
        ns=torch.cat([(no*next_mask[:,:,None]).flatten(1),next_mask],-1)
        with torch.no_grad():
            joint=torch.cat([actor(no[:,i])*next_mask[:,i,None] for i,actor in enumerate(self.target_actors)],-1)
            targets=[r[:,i]+c.gamma*(1-t[:,i])*critic(torch.cat([ns,joint],-1)).squeeze(-1)
                     for i,critic in enumerate(self.target_critics)]
        losses={}
        for i,agent in enumerate(self.ids):
            valid=mask[:,i];denom=valid.sum().clamp_min(1)
            if not valid.any():continue
            critic=self.critics[i];opt=self.critic_opts[i]
            q=critic(torch.cat([s,a.flatten(1)],-1)).squeeze(-1)
            loss=((q-targets[i]).square()*valid).sum()/denom
            opt.zero_grad();loss.backward();opt.step();opt.zero_grad(set_to_none=True)
            critic.requires_grad_(False)
            try:
                joint=torch.cat([self.actors[i](o[:,i]) if j==i else a[:,j].detach() for j in range(len(self.ids))],-1)
                actor_loss=-(critic(torch.cat([s,joint],-1)).squeeze(-1)*valid).sum()/denom
                self.actor_opts[i].zero_grad();actor_loss.backward();self.actor_opts[i].step()
            finally:critic.requires_grad_(True)
            losses[agent]={'actor':float(actor_loss.detach()),'critic':float(loss.detach()),'q_mean':float((q.detach()*valid).sum()/denom)}
        with torch.no_grad():
            for online,target in [(self.actors,self.target_actors),(self.critics,self.target_critics)]:
                for x,y in zip(online.parameters(),target.parameters()):y.lerp_(x,c.tau)
        self.updates+=1
        if not all(np.isfinite(v) for d in losses.values() for v in d.values()):raise FloatingPointError('nonfinite losses')
        return losses

    def save(self,path,metadata):
        torch.save(dict(critic_encoding_version=2,config=asdict(self.config),ids=self.ids,steps=self.steps,updates=self.updates,
            actors=self.actors.state_dict(),critics=self.critics.state_dict(),target_actors=self.target_actors.state_dict(),
            target_critics=self.target_critics.state_dict(),actor_opts=[x.state_dict() for x in self.actor_opts],
            critic_opts=[x.state_dict() for x in self.critic_opts],replay=list(self.replay),rng=self.rng.bit_generator.state,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),metadata=metadata),path)

    def restore_rng(self):
        """Explicit training continuation only; inference must not reset global RNGs."""
        random.setstate(self.saved_rng['python_rng'])
        np.random.set_state(self.saved_rng['numpy_rng'])
        torch.set_rng_state(self.saved_rng['torch_rng'])

    @classmethod
    def load(cls,path,restore_rng=False):
        d=torch.load(path,map_location='cpu',weights_only=False);m=cls(d['ids'],MADDPGConfig(**d['config']))
        for name in ['actors','critics','target_actors','target_critics']:getattr(m,name).load_state_dict(d[name])
        for name in ['actor_opts','critic_opts']:
            for opt,state in zip(getattr(m,name),d[name]):opt.load_state_dict(state)
        m.replay.extend(d['replay']);m.steps=d['steps'];m.updates=d['updates'];m.rng.bit_generator.state=d['rng']
        m.replay_compatible=d.get('critic_encoding_version')==2
        if restore_rng and not m.replay_compatible:raise ValueError('Historical replay encoding cannot continue training')
        m.metadata=d['metadata'];m.saved_rng={k:d[k] for k in ['python_rng','numpy_rng','torch_rng']}
        if restore_rng:m.restore_rng()
        return m
