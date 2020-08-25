import sys
import numpy as np
import matplotlib.pyplot as plt 

region = sys.argv[1]

fleet_risk_1x = np.loadtxt(region+'_1x_fleet_hourly_risk.csv')
generator_risk_1x = np.loadtxt(region+'_1x_generator_hourly_risk.csv')
generator_contribution_1x = generator_risk_1x - fleet_risk_1x
final_risk_1x = np.loadtxt(region+'_1x_renewables_hourly_risk.csv')
final_risk_diff_1x = final_risk_1x - fleet_risk_1x

fleet_risk_2x = np.loadtxt(region+'_2x_fleet_hourly_risk.csv')
generator_risk_2x = np.loadtxt(region+'_2x_generator_hourly_risk.csv')
generator_contribution_2x = generator_risk_2x - fleet_risk_2x
final_risk_2x = np.loadtxt(region+'_2x_renewables_hourly_risk.csv')
final_risk_diff_2x = final_risk_2x - fleet_risk_2x

print('1X RENEWABLES:\n')

generator_contribution_hours = np.argwhere(generator_contribution_1x != 0).flatten()

print(generator_contribution_hours)
print(generator_contribution_1x[generator_contribution_hours])
print(np.sum(generator_contribution_1x))

risk_change_hours = np.argwhere(final_risk_diff_1x != 0).flatten()
print(risk_change_hours)
print(final_risk_diff_1x[risk_change_hours])

print('2X RENEWABLES:\n')

generator_contribution_hours = np.argwhere(generator_contribution_2x != 0).flatten()

print(generator_contribution_hours)
print(generator_contribution_2x[generator_contribution_hours])
print(np.sum(generator_contribution_2x))

risk_change_hours = np.argwhere(final_risk_diff_2x != 0).flatten()
print(risk_change_hours)
print(final_risk_diff_2x[risk_change_hours])

# plot

start = 3000
end = 6000

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(generator_contribution_1x,alpha=.5)
ax.plot(generator_contribution_2x,alpha=.5)
ax.legend(['1x Existing Renewables', '2x Existing Renewables'])
ax.set_title('$\Delta$ Risk W/ Generator Contribution\n'+region)
ax.set_xlabel('Hour of Year')
ax.set_ylabel('$\Delta$ LOLP')
ax.set_xlim([start,end])

plt.savefig('added_reliability_'+region)