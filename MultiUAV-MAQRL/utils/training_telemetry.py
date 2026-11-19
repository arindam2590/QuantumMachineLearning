"""Read-only algorithm-side per-episode telemetry, with bounded memory."""
from collections import defaultdict
import numpy as np
import torch


class EpisodeTelemetry:
    def __init__(self,agent_ids,initial_distances):
        self.ids=tuple(agent_ids)
        self.initial=dict(initial_distances);self.minimum=dict(initial_distances);self.final=dict(initial_distances)
        self.actions={a:np.zeros(3) for a in self.ids};self.saturation={a:np.zeros(3) for a in self.ids}
        self.negative_saturation={a:np.zeros(3) for a in self.ids}
        self.counts=defaultdict(int);self.components={a:defaultdict(float) for a in self.ids}
        self.status={a:'active' for a in self.ids};self.q={a:[] for a in self.ids}
        self.loss={a:defaultdict(list) for a in self.ids}

    def observe(self,actions,infos,losses=None):
        for a,value in actions.items():
            value=np.asarray(value)
            self.negative_saturation[a]+=value<=-.98
            self.actions[a]+=value;self.saturation[a]+=np.abs(value)>=.98;self.counts[a]+=1
            self.final[a]=float(infos[a]['distance_to_goal']);self.minimum[a]=min(self.minimum[a],self.final[a])
            self.status[a]=infos[a]['status']
            for name,v in infos[a]['reward_components'].items():self.components[a][name]+=float(v)
        for a,values in (losses or {}).items():
            for name,v in values.items():self.loss[a][name].append(float(v))

    def sample_q(self,learner,adapter,actions):
        critics=getattr(getattr(learner,'networks',None),'critics',None)
        if critics is None:return
        state=torch.as_tensor(adapter.centralized_state(),device=learner.device).unsqueeze(0)
        joint=np.concatenate([actions.get(a,np.zeros(3,np.float32)) for a in self.ids])
        joint=torch.as_tensor(joint,dtype=torch.float32,device=learner.device).unsqueeze(0)
        with torch.no_grad():
            for i,a in enumerate(self.ids):
                if a in actions:self.q[a].append(float(critics[i](state,joint).item()))

    def result(self):
        return dict(action_mean={a:(self.actions[a]/max(1,self.counts[a])).tolist() for a in self.ids},
            negative_action_saturation={a:(self.negative_saturation[a]/max(1,self.counts[a])).tolist() for a in self.ids},
            action_saturation={a:(self.saturation[a]/max(1,self.counts[a])).tolist() for a in self.ids},
            initial_destination_distance=self.initial,minimum_destination_distance=self.minimum,
            final_destination_distance=self.final,destination_progress={a:self.initial[a]-self.final[a] for a in self.ids},
            agent_status=self.status,failure_reasons={a:s for a,s in self.status.items() if s not in ('active','arrived','truncated')},
            reward_components={a:dict(v) for a,v in self.components.items()},
            mean_losses={a:{k:float(np.mean(v)) for k,v in terms.items()} for a,terms in self.loss.items()},
            critic_q={a:dict(samples=len(v),mean=float(np.mean(v)) if v else None,
                minimum=min(v) if v else None,maximum=max(v) if v else None) for a,v in self.q.items()})
