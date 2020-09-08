from utils import import_models
import_models()
from raostar import RAOStar
from ashkan_icaps_model import *
from iterative_raostar import *
from icaps_sim import *
import numpy as np
import sys
import graph_to_json

if __name__ == '__main__':
    # min_distance = 1
    # print('risk', risk_for_two_gaussians(1, 0.5, 2.5, 0.5, min_distance))

    # default cc value
    cc = 0.4
    if len(sys.argv) > 1:
        cc = float(sys.argv[1])

    model = Ashkan_ICAPS_Model(str(cc * 100) + "% risk")
    algo = RAOStar(model, cc=cc, debugging=False,
                   cc_type='e', ashkan_continuous=True)

    agent_coords = [6, 9]
    agent_coords = [7, 3]

    b0 = ContinuousBeliefState(9, 1)
    b0.set_agent_coords(agent_coords[0], agent_coords[1])
    P, G = algo.search(b0)

    most_likely_policy(G, model)
    Sim = Simulator(10, 10, G, P, model, grid_size=50)

    # Convert all matrices to strings for json
    # print(G)

    gshow = graph_to_json.policy_to_json(G, 0.5, 'results/ashkan_9_1.json')

    # code to draw the risk grid
    # for i in range(11):
    #     for j in range(11):
    #         static_risk = static_obs_risk_coords(
    #             np.matrix([i, j]), np.matrix([[0.2, 0], [0, 0.2]]))
    #         dynamic_risk = dynamic_obs_risk_coords(
    #             np.matrix([i, j]), np.matrix([[0.2, 0], [0, 0.2]]), np.matrix(agent_coords), np.matrix([[0.5, 0], [0, 0.5]]), min_distance)
    #         risk = max(static_risk, dynamic_risk)
    #         Sim.draw_risk_colors(i, j, risk)
    # print(i, j,risk)

    Sim.start_sim()

    # print(V, V2)
    # print(sum(np.multiply(V, V2)))
    #
    # V = [np.float64(3104849510969301127.0) / np.float64(18014398509481984),
    #      np.float64(-1503248202618249621.0) / np.float64(18014398509481984),
    #      np.float64(22975721368860239.0) / np.float64(562949953421312),
    #      -363604901253686789.0 / 18014398509481984,
    #      182377528050986903.0 / 18014398509481984,
    #      -92606084735608187.0 / 18014398509481984,
    #      23835889479366529.0 / 9007199254740992,
    #      -25337103963040591.0 / 18014398509481984,
    #      12991712941642483.0 / 18014398509481984,
    #      -6220965932913649.0 / 18014398509481984,
    #      551428736999639.0 / 2251799813685248,
    #      -2838174069571369.0 / 18014398509481984,
    #      570470459930619.0 / 18014398509481984,
    #      -3095564674573975.0 / 18014398509481984,
    #      1992285035488617.0 / 9007199254740992,
    #      6369051672525773.0 / 18014398509481984,
    #      -11494468648281361.0 / 18014398509481984,
    #      -4761164806390477.0 / 18014398509481984,
    #      1]
    #
    # V2 = [np.float64(17.0 / 3566),
    #       np.float64(-109.0 / 5146),
    #       np.float64(97.0 / 5452),
    #       35.0 / 951,
    #       -57.0 / 883,
    #       130.0 / 9007,
    #       97.0 / 5090,
    #       254.0 / 9893,
    #       -41.0 / 7464,
    #       -628.0 / 5187,
    #       213.0 / 1510,
    #       703.0 / 47980,
    #       -306.0 / 5065,
    #       -195.0 / 3889,
    #       -265.0 / 3987,
    #       755.0 / 1752,
    #       -1031.0 / 2802,
    #       -634.0 / 1917,
    #       295.0 / 747]
    #
    # print(sum(np.multiply(V, V2)))
