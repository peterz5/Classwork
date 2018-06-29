import sys
import numpy as np


if len(sys.argv) != 4:
	print('invalid input!')
	sys.exit(0)

else:
	try:
		N = 400

		T = int(sys.argv[1])
		score1 = int(sys.argv[2])
		score2 = int(sys.argv[3])

		T1 = T-score1
		T2 = T-score2

		n = 0
		p = [0]*7
		Max = 0
		turn = 0


		even = [[0.0 for row in range(T2+1)] for col in range(T1+1)]
		odd = [[0.0 for row in range(T2+1)] for col in range(T1+1)]

		turntotals = [[0.0 for col in range(T2+1)] for row in range(T1+1)]

	#n = 0 matrix

		#all ending positons 
		for i in range(T1):
			for j in range(T2):
				even[i][j] = .5

		#last row
		for i in range(T1):
			even[i][T2] = 0

		#last column
		for i in range(T2+1):
			even[T1][i] = 1

		#if n >0

		if T > 6:
			p = [0] * (T+1)
		
		nobust = 5/6

		#probability of getting something
		p[0] = 1
		p[1] = 0
		p[2] = 1/6
		p[3] = 1/6
		p[4] = 1/6 + p[2] * 1/6
		p[5] = 1/6 + p[2] * 1/6 + p[3]* 1/6
		p[6] = 1/6 + p[3] * 1/6 + p[2]*1/6 + p[4]*1/6
		
		
		for i in range(7, T+1): #generate probabilities for all numbers up through T
			for j in range(i-6, i-1): #previous 5 numbers
				for k in range(2, 7): #2 - 6
					if j+k == i:
						p[i] += p[j]*1/6 


		#generates probabilities of going for a number and landing on total
		prob = [[0 for x in range(T+6)] for y in range(T+2)]

		for i in range (T+1):
			q = 0
			for j in range(i, i+6):
				for k in range(1, min(i+1, i-j+7)):
					if not(i == j and k == 1):
						prob[i][j] += p[i-k] * 1/6
				q += prob[i][j] 
			prob[i][0] = 1-q

		#while n less than turn limit or maxes converge at 10 digits
		prevMax = 0
		while n < N and (round(even[0][0], 10) != round(prevMax, 10)):
			n += 1

			if n % 2 == 0:
				prevMax = even[0][0]
				for j in range(0, T2):
					for x in range(0, T1+1): # for every possible score
						Max = 0
						for s in range(2, max(2, T1-x)+1): #searching for optimal roll
							
							val = 0
							ev = 0
							tp = 0

							for k in range(s, s+6):
								val += odd[min(T1,x+k)][j]*prob[s][k]
								tp += prob[s][k]


							bust = 1 - tp
							ev = val + bust * odd[x][j] #expected value


							if ev > Max:
								Max = ev 
								turntotals[x][j] = s #best thing to roll to
		
						even[x][j] = Max #chance of winning after rolling to this score		
			else:

				for i in range(0, T1+1):
					for y in range(0, T2):#possible adversary scores
						Min = 1000

						for s in range(2, max(2, T2-y) +1): #possible scores adversary can roll to
							val = 0
							ev = 0
							tp = 0

							for k in range(s, s+6):
								val += even[i][min(T2, y+k)]*prob[s][k]
								tp += prob[s][k]
							bust = 1 - tp

							ev = val + bust * even[i][y] #adversary expected value
							
							if ev < Min:
								Min = ev

						odd[i][y] = Min #Want adversary to have min chance of winning

		print(round(even[0][0], 6), turntotals[0][0])

	except:
		print('invalid input!')
		sys.exit(0)


