import numpy as np

from datasets.hypothesis_routes import dynamic_candidate_mask


def test_dynamic_candidate_mask_respects_max_shape_and_choices():
    config = dict(goal_top_k=64, exploration_candidates=8,
                  dynamic_goal_top_k=[8, 16, 32, 48, 64],
                  dynamic_exploration_candidates=[1, 2, 4, 8],
                  hypothesis_tail_train_fraction=0.0)
    seen=set()
    for digest in range(20):
        candidates=dict(candidate_mask=np.ones(72,dtype=bool))
        k,e,offset=dynamic_candidate_mask(candidates,config,digest)
        assert offset==0
        assert candidates['candidate_mask'][:64].sum()==k
        assert candidates['candidate_mask'][64:].sum()==e
        seen.add((k,e))
    assert seen=={(k,e) for e in (1,2,4,8) for k in (8,16,32,48,64)}


def test_dynamic_candidate_mask_can_select_refresh_tail():
    config = dict(goal_top_k=64, exploration_candidates=8,
                  dynamic_goal_top_k=[16],dynamic_exploration_candidates=[2],
                  hypothesis_tail_train_fraction=1.0)
    candidates=dict(candidate_mask=np.ones(72,dtype=bool))
    k,e,offset=dynamic_candidate_mask(candidates,config,123456)
    assert (k,e)==(16,2)
    assert offset!=0
    assert candidates['candidate_mask'].sum()==18
