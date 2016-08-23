from scipy import stats

def main():
	"""
	1st phase
	top1 = [70.0, 71.1, 72.5, 70.8, 68.1, 71.9, 71.1, 71.3, 68.4, 70.2]
	top3 = [75.8, 78.4, 77.8, 77.7, 80.0, 77.8, 78.7, 76.4, 79.1, 77.3]
	2nd phase
	"""
	x = [53.6, 54.5, 53.7, 52.7, 53.1, 55.5, 55.5, 52.8, 53.7, 52.7]
	y = [89.7, 89.1, 89.5, 88.7, 89.4, 88.6, 89.8, 89.5, 89.2, 89.7]
	# Compute the Wilcoxon rank-sum statistic for two samples.
	wilcoxon = stats.ranksums(x, y)
	anova = stats.f_oneway(x, y)
	print "Wilcoxon: " + str(wilcoxon[1]) + "; ANOVA: " + str(anova[1])

if __name__ == '__main__':
	main()