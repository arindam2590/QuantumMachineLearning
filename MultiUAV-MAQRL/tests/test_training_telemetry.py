import numpy as np
from utils.training_telemetry import EpisodeTelemetry


def test_episode_telemetry_aggregation():
    t=EpisodeTelemetry(['uav_0'],{'uav_0':100.})
    t.observe({'uav_0':np.array([-1.,0.,1.])},{'uav_0':dict(distance_to_goal=102.,status='out_of_bounds',reward_components={'progress':-.2})},
              {'uav_0':{'actor_loss':1.,'critic_loss':2.}})
    r=t.result()
    assert r['action_mean']['uav_0']==[-1,0,1]
    assert r['action_saturation']['uav_0']==[1,0,1]
    assert r['destination_progress']['uav_0']==-2
    assert r['minimum_destination_distance']['uav_0']==100
    assert r['failure_reasons']=={'uav_0':'out_of_bounds'}
    assert r['mean_losses']['uav_0']['critic_loss']==2
    assert r['critic_q']['uav_0']['samples']==0
