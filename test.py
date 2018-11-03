import sys
import utils
import csv
import pprint
import ast

pp = pprint.PrettyPrinter(width=41)

def run_test_suite(benchmark_filepath, test_scans):
	check_for_errors(benchmark_filepath, test_scans)

	benchmark_scans = utils.extractCSVtoDict(benchmark_filepath)
	benchmark_ids = [voter_id for voter_id in benchmark_scans]

	results = {'correct_question_scans': 0, 'false_positive_question_scans': 0, 'false_negative_question_scans': 0}

	for scan in test_scans:
		print('-- {} --'.format(scan['voter_id']))
	
		# check if the scan is in the benchmark
		if not scan['voter_id'] in benchmark_ids:
			print('Voter ID {} not in benchmark, skipping.'.format(scan['voter_id']))
			continue

		benchmark_scan = benchmark_scans[scan['voter_id']]
		benchmark_questions = benchmark_scan['questions']
		questions = scan['questions']

		# due to how results CVS is generated, should fix this
		if 'voter_id' in questions:
			questions.pop('voter_id')
		if 'voter_id' in benchmark_questions:
			benchmark_questions.pop('voter_id')

		for question, responses in questions.items():
			# print(question)

			benchmark_responses = utils.convertStringListToList(benchmark_questions[question])
			test_responses = utils.convertStringListToList(responses)

			# print(benchmark_responses)
			# print(test_responses)

			for response in test_responses:
				if response in benchmark_responses:
					benchmark_responses.remove(response)
					test_responses.remove(response)

			# check if there are remaining values
			if not benchmark_responses and not test_responses:
				results['correct_question_scans'] += 1
				print('correct scan')
			elif benchmark_responses:
				results['false_negative_question_scans'] += 1
				print('false negative')
			elif test_responses:
				print('false Positive')
				results['false_positive_question_scans'] += 1

			# drop the question from the benchmark
			benchmark_questions.pop(question)

		# check if there are non-empty benchmark questions that weren't caught
		benchmark_questions = {k:v for k,v in benchmark_questions.items() if v is not ''}
		if benchmark_questions:
			print(len(benchmark_questions))
			results['false_negative_question_scans'] += len(benchmark_questions)
		else:
			results['correct_question_scans'] += 1
			print('correct scan')

	show_statistics(results)


def check_for_errors(benchmark_filepath, test_scans):
	# Check that the benchmark file exists
	try:
			fh = open(benchmark_filepath, 'r')
	except FileNotFoundError:
		print ("Benchmark file not found")
		sys.exit()

	# Check if there are test scans
	if len(test_scans) == 0:
		print('No test scans')
		sys.exit()


def show_statistics(results):
	print('================ BENCHMARK STATISTICS ================')
	
	total_question_scans = results['correct_question_scans'] + results['false_negative_question_scans'] + results['false_positive_question_scans']

	# calc rates
	correct_rate = round(results['correct_question_scans'] / total_question_scans * 100)
	false_positive_rate = round(results['false_positive_question_scans'] / total_question_scans * 100)
	false_negative_rate = round(results['false_negative_question_scans'] / total_question_scans * 100)
	print('Correct Question Scans: {} of {}, {}%'.format(results['correct_question_scans'], total_question_scans, correct_rate))
	print('False Negative Question Scans: {} of {}, {}%'.format(results['false_negative_question_scans'], total_question_scans, false_negative_rate))
	print('False Positive Question Scans: {} of {}, {}%'.format(results['false_positive_question_scans'], total_question_scans, false_positive_rate))












