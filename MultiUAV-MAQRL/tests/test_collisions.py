from utils.geometry import aabb_distance

def test_aabb_distance():
    assert aabb_distance([2,0,0], [0,0,0], [1,1,1]) == 1
