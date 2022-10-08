from random import random
import random

li = [x * random.randint(1,10) for x in range(10)]
li.append(10)
li.append(10)

# Remove duplicates
li_unique = set(li)
print(len(li_unique))
li_reversed = reversed(li)
print(li_reversed)