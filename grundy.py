import sys

#find the mininmum excluded number of any list of numbers
#iterate through until you get all the way through without a member of the list 
#equaling minex, if that happens you have your minex
def mex(lst):
	
	minex = 0
	traversed = False

	while traversed != True:
		traversed = True
		for i in range(len(lst)):
			if lst[i] == minex:
				traversed = False
				minex += 1
				break

	return minex

#finds the grundy number for n, and also updates the grundynumbers table in the 
#process so we don't have to recalculate values
def grundy(n):
	if(grundynumbers[n] == None): # if we haven't seen this before
		if n == 0:
			grundynumbers[n] = 0 #base case, update table
		elif n == 1:
			grundynumbers[n] = 1 #base case
		elif n == 2:
			grundynumbers[n] = 0 #base case
		else:
			arraytomex = []

			array = [1] * n
			for i in range(n-2): #create pin representation
				array[i] = 0
				array[i+1] = 0

				count = 0
				lst = []

				for j in range(len(array)): #count consecutive pins then add to list
					if array[j] == 0:
						if count != 0:
							lst.append(count)
							count = 0
					else:
						count +=1
				if count != 0:
					lst.append(count)

				nimber = 0
				for k in lst:
					nimber = nimber ^ grundy(k) #XOR is commutative, XOR to find grundy number 
				arraytomex.append(nimber) #to be added to array and mexed

				array[i] = 1
			grundynumbers[n] = mex(arraytomex) #update table with new value
	
	return grundynumbers[n] #return grundy number for n

#method to break apart and analyze the pin representation
def decipher(s):
	
	lst = []
	count = 0
	for i in s:
		if i == '.':
			if count != 0: #if we have seen some x's before, add them to our list
				lst.append(count)
				count = 0
		else:
			count +=1
	if count != 0: #if we run off the end of the string before seeing another '.'
		lst.append(count)
	nimber = 0
	for k in lst:
		nimber = grundy(nimber) ^ grundy(k)
	if nimber <= 0: #a non-positive grundy number indicates loss
		return ('LOSS')

	else: #if n position then we need to show move to p position
			copy = ''
			for i in range(len(s)):
				if s[i] == 'x' and (i == len(s)-1 or s[i+1] == '.') and (i==0 or s[i-1] == '.'):
					copy = s[:i] + '.' + s[i+1:]
					if (decipher(copy) == 'LOSS'):
						return copy


#if given the empty set, that is a p position and you lose
if len(sys.argv) == 1:
	print('LOSS') 
	
elif len(sys.argv) == 2:
	for i in sys.argv[1]:
		if i != 'x' and i != '.':
			print('wrong input!')
			exit(0)

	#initialize table
	grundynumbers = [None] * (len(sys.argv[1])+1)
	
	#if given string results in a loss, print loss otherwise indicate winning move
	result = decipher(sys.argv[1])
	print(result)
	
elif len(sys.argv) == 3:
	#number to find the grundy of n

	if sys.argv[1] != 'grundy':
		print('wrong input!')
		exit(0)
	try:
		int(sys.argv[2])
		n = int(sys.argv[2])
	except ValueError:
		print('You\'re argument was not a number!')
		exit(0)

	#initialize table
	grundynumbers = [None] * (n+1)

	#print list of grundy numbers
	for i in range(n+1):
		grundynumbers[i] = grundy(i)
	print(grundynumbers)

else:
	print('Too many arguments!')
	exit(0)