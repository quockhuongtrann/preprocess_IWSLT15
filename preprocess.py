from vncorenlp import VnCoreNLP
import re
annotator = VnCoreNLP("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g') 

def process_double_quotes(text): #text is a string
	regex_list = re.findall("\"\s(.*?)\s\"", text)
	regex_list_old = []
	regex_list_new = []
	for i in range(len(regex_list)):
		regex_list_old.append("\" " + regex_list[i] + " \"")
		regex_list_new.append("\"" + regex_list[i] + "\"")
	for i in range(len(regex_list)):
		text = text.replace(regex_list_old[i], regex_list_new[i])
	return text

def process_single_quotes(text): #text is string
	regex_list = re.findall("'\s(.*?)\s'", text)
	regex_list_old = []
	regex_list_new = []
	for i in range(len(regex_list)):
		regex_list_old.append("' " + regex_list[i] + " '")
		regex_list_new.append("'" + regex_list[i] + "'")
	for i in range(len(regex_list)):
		text = text.replace(regex_list_old[i], regex_list_new[i])
	return text

def convert_list_to_string(text): #text is a list
	flatten_text = [item for sublist in text for item in sublist]
	result = ' '.join(item for item in flatten_text)
	result = result.replace("- -", "--")
	result = result.replace(" - ", "-")
	result = result.replace(" .",".")
	result = result.replace(" ,",",")
	result = result.replace(" ?",",")
	result = result.replace(" !",",")
	result = result.replace(" :",",")
	result = result.replace(" ;",",")
	result = result.replace("[ ", "[")
	result = result.replace(" ]", "]")
	result = result.replace("( ", "(")
	result = result.replace(" )", ")")
	result = result.replace("{ ", "{")
	result = result.replace(" }", "}")
	result = process_single_quotes(result)
	result = process_double_quotes(result)
	# result = process_square_bracket(result)
	return result

def preprocess_vi(filename, out_filename):
	with open(filename, 'r', encoding='utf-8') as f:
		all_docs = f.readlines()
	wseg_texts = []
	for sent in all_docs:
		wseg_text = annotator.tokenize(sent)
		wseg_texts.append(convert_list_to_string(wseg_text))
	with open(out_filename, 'w', encoding='utf-8') as fo:
		for line in wseg_texts:
			# if(line != ""):
				print(line.strip(), file=fo)
	# return wseg_texts

preprocess_vi(
	"/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/preprocess_data/test.en-vi.vi",
	"/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/test.en-vi.vi")
print("Test Data: Done.")
preprocess_vi(
	"/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/preprocess_data/valid.en-vi.vi",
	"/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.vi")
print("Valid Data: Done.")
preprocess_vi(
	"/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/preprocess_data/train.en-vi.vi",
	"/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/train.en-vi.vi")
print("Train Data: Done.")
# print(wseg_texts[-1])
