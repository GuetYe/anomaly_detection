import matplotlib.pyplot as plt
from processed.IBRL_load import get_IBRL_data
from Config import parse_args

config = parse_args()
data = get_IBRL_data()

# inject_modal=1
# inject_node=38
# start_time=1844
# end_time=1849
# reference_modal=0
#
# inter_modal = True
# inter_node = False
#
# if inter_modal == True:
#     s1 = data[inject_modal][inject_node][start_time - config.inject_length
#                                          :end_time + config.window_size - config.inject_length]
#     s2 = data[reference_modal][inject_node][start_time - config.inject_length
#                                          :end_time + config.window_size - config.inject_length]
#     plt.plot([i for i in range(len(s1))], s1)
#     plt.plot([i for i in range(len(s2))], s2)
#     plt.show()


# inject_modal=0
# inject_node=28
# start_time=856
# end_time=861
# reference_modal=1
# reference_node=0
#
# inter_modal = False
# inter_node = True
#
# if inter_node == True:
#     s1 = data[inject_modal][inject_node][start_time - config.inject_length
#                                          :end_time + config.window_size - config.inject_length]
#     s2 = data[inject_modal][reference_node][start_time - config.inject_length
#                                          :end_time + config.window_size - config.inject_length]
#     plt.plot([i for i in range(len(s1))], s1)
#     plt.plot([i for i in range(len(s2))], s2)
#     plt.show()