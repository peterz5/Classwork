Pig:
	echo "#!/bin/bash" > Pig
	echo "python3 pig.py \"\$$@\"" >> Pig
	chmod u+x Pig