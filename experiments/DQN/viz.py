import pandas as pd
import matplotlib.pyplot as plt

agents25_original = pd.read_pickle("DQN_without_fix")
agents50_fix = pd.read_pickle("DQN_WITH_fix_50_Agents")

m = agents25_original.mean()
s = agents25_original.std()

episode = pd.Series(range(0, agents25_original.shape[1]))

plt.plot(episode, m)
plt.fill_between(episode, m - s, m + s, facecolor='r')
plt.ylabel('Average return')
plt.xlabel('Episode')
plt.ylim([0, 200])
plt.grid()
plt.title('25 Agent Original')
plt.show()

m_fix = agents50_fix.mean()
s_fix = agents50_fix.std()

episode_fix = pd.Series(range(0, agents50_fix.shape[1]))

plt.plot(episode_fix, m_fix)
plt.fill_between(episode_fix, m_fix - s_fix, m_fix + s_fix, facecolor='r')
plt.ylabel('Average return')
plt.xlabel('Episode')
plt.ylim([0, 200])
plt.grid()
plt.title('50 Agent Fixed Code')
plt.show()
