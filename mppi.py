import ur5_env as env

ur5 = env.ur5()
ur5.reset()

obs, reward, done, term, info = ur5.step(ur5.action_space.sample())
print(obs)
