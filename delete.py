import time

line = ""
with open("delete_F_names","r") as f:
	line = f.readline()


line = line.split(",")

print(len(line))
final_ans = [] 

for i in line:
	final_ans = final_ans + ["AVG(" + i + ") AS AVG_" + i] + ["STDEV(" + i + ") AS STDEV_" + i]


print(len(final_ans))

writer = open("delete","w")

for i in final_ans:
	writer.write(i+"\n")

