import csv
import re

from_fhdl = open('answers.csv')
to_fhdl = open('Base_answers_formatted.csv', 'w')

csv.register_dialect('answer', delimiter=',', quoting=csv.QUOTE_NONE, escapechar = '"')

csv_writer = csv.writer(to_fhdl, dialect = 'answer')

result = []
for line in from_fhdl:
	data = line.strip().split(',')
	print data
	que_id = data[0]

	que = ''.join(data[1:]).strip(',')
	result.append([que_id, que])
	
csv_writer.writerows(result)

print "end"


