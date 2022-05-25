import argparse
import os
import xmltodict
import glob
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
	return result

def extract_train(train_inp, train_out, docids_out=None, vi=False):
    with open(train_inp, "r", encoding="utf-8") as f:
        all_docs = f.readlines()
        print(len(all_docs))
    delete_tags = ["</url>", "</keywords>", "</speaker>", "</title>", 
                "</description>", "</reviewer>", "</translator>"]
    remove_text = []
    content_text = []
    for line in all_docs:
        if any(tag in line for tag in delete_tags):
            remove_text.append(line)
        # elif line =="\n":
        #     remove_text.append(line)
        else:
            content_text.append(line)
    idx_id = []
    for i in range(len(content_text)):
        if "</talkid>" in content_text[i]:
            idx_id.append(i)
    print(len(content_text))
    for idx in idx_id:
        content_text[idx] = content_text[idx].replace("<talkid>", "")
        content_text[idx] = content_text[idx].replace("</talkid>\n", "")
    
    docs_dict = []
    for i in range(len(idx_id)-1):
        docs_info = {"doc_id": content_text[idx_id[i]],
                    "texts": []}
        for j in range(idx_id[i]+1, idx_id[i+1]):
            docs_info["texts"].append(content_text[j])
        docs_dict.append(docs_info)

    train_out_f = open(train_out, "a", encoding="utf-8")
    if docids_out is not None:
        docids_out_f = open(docids_out, "a", encoding="utf-8")

    for doc in docs_dict:
        doc_id = doc["doc_id"]
        if vi == False:
            for line in doc["texts"]:
                print(line.strip(), file=train_out_f)
                if docids_out is not None:
                    print(doc_id, file=docids_out_f)
        if vi == True:
            wseg_texts = []
            for sent in doc["texts"]:
                wseg_text = annotator.tokenize(sent)
                wseg_texts.append(convert_list_to_string(wseg_text))
            for line in wseg_texts:
                print(line.strip(), file=train_out_f)
                if docids_out is not None:
                    print(doc_id, file=docids_out_f)

# def extract_content_from_doc(doc):
#     lines = doc.split('\n')
#     delete_tags = ["</url>", "</keywords>", "</speaker>", "</title>", 
#                 "</description>", "</reviewer>", "</translator>"]
#     remove_text = []
#     content_text = []
#     for line in lines:
#         if any(tag in line for tag in delete_tags):
#             remove_text.append(line)
#         # elif line == "\n":
#         #     remove_text.append(line)
#         else:
#             content_text.append(line)
#     content_text[0] = content_text[0].replace("<talkid>", "")
#     content_text[0] = content_text[0].replace("</talkid>", "")
#     return {
#         "doc_id": content_text[0],
#         "texts": content_text[1:]
#     }
        

# def extract_train(train_inp, train_out, docids_out=None, vi=False):
#     with open(train_inp, 'r', encoding='utf-8') as f:
#         all_docs = f.read()
#         docs_xml = [f"{d}</translator>" for d in all_docs.split("</translator>")[:-1]]
#     docs_dict = []
#     for doc in docs_xml:
#         docs_dict.append(extract_content_from_doc(doc))
#     # print(docs_dict[100])
#     train_out_f = open(train_out, "a", encoding="utf-8")
#     if docids_out is not None:
#         docids_out_f = open(docids_out, "a", encoding="utf-8")

#     for doc in docs_dict:
#         doc_id = doc["doc_id"]
#         if vi == False:
#             for line in doc["texts"]:
#                 print(line.strip(), file=train_out_f)
#                 if docids_out is not None:
#                     print(doc_id, file=docids_out_f)
#         if vi == True:
#             wseg_texts = []
#             for sent in doc["texts"]:
#                 wseg_text = annotator.tokenize(sent)
#                 wseg_texts.append(convert_list_to_string(wseg_text))
#             for line in wseg_texts:
#                 print(line.strip(), file=train_out_f)
#                 if docids_out is not None:
#                     print(doc_id, file=docids_out_f)

def extract_eval(eval_inp, eval_out, docids_out=None, vi=False):
    with open(eval_inp, "r", encoding="utf-8") as f:
        all_docs = f.read()
        docs_xml = xmltodict.parse(all_docs)
        docs = (
            docs_xml["mteval"]["srcset"]["doc"]
            if "srcset" in docs_xml["mteval"]
            else docs_xml["mteval"]["refset"]["doc"]
        )

    eval_out_f = open(eval_out, "a")
    if docids_out is not None:
        docids_out_f = open(docids_out, "a")

    if vi == True:
        for doc in docs:
            doc_id = doc["@docid"]
            if not doc["seg"]:
                continue
            wseg_texts = []
            for seg in doc["seg"]:
                line = seg["#text"]
                wseg_text = annotator.tokenize(line)
                wseg_texts.append(convert_list_to_string(wseg_text))
            for line in wseg_texts:
                print(line.strip(), file=eval_out_f)
                if docids_out is not None:
                    print(doc_id, file=docids_out_f)
    if vi == False:
        for doc in docs:
            doc_id = doc["@docid"]
            if not doc["seg"]:
                continue
            for seg in doc["seg"]:
                line = seg["#text"]
                print(line.strip(), file=eval_out_f)
                if docids_out is not None:
                    print(doc_id, file=docids_out_f)

#TRAIN
extract_train("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/train.tags.en-vi.en",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/train.en-vi.en",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/train.en-vi.docids",
            )

extract_train("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/train.tags.en-vi.vi",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/train.en-vi.vi",
            vi=True,
            )

#VALID
extract_eval("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/IWSLT15.TED.tst2010.en-vi.en.xml",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.en",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.docids")

extract_eval("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/IWSLT15.TED.tst2010.en-vi.vi.xml",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.vi",
            vi=True,)

extract_eval("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/IWSLT15.TED.tst2011.en-vi.en.xml",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.en",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.docids")

extract_eval("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/IWSLT15.TED.tst2011.en-vi.vi.xml",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.vi",
            vi=True,)

extract_eval("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/IWSLT15.TED.tst2012.en-vi.en.xml",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.en",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.docids")

extract_eval("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/IWSLT15.TED.tst2012.en-vi.vi.xml",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/valid.en-vi.vi",
            vi=True,)

#TEST
extract_eval("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/IWSLT15.TED.tst2013.en-vi.en.xml",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/test.en-vi.en",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/test.en-vi.docids")

extract_eval("/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data15/IWSLT15.TED.tst2013.en-vi.vi.xml",
            "/home/tqkhuong/Documents/THESIS/preprocessIWSLT15/data_wseg/test.en-vi.vi",
            vi=True,)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("parser for iwslt2017 data")

#     parser.add_argument("raw_data")
#     parser.add_argument("out_data")
#     parser.add_argument("-s", "--source-lang", required=True, type=str)
#     parser.add_argument("-t", "--target-lang", required=True, type=str)
#     parser.add_argument(
#         "--devsets", nargs="*", default=["tst2010", "tst2011", "tst2012"]
#     )
#     parser.add_argument("--validsets", nargs="*", default=["tst2013"])
#     args = parser.parse_args()

#     first_l = (
#         args.source_lang
#         if glob.glob(f"{args.raw_data}/*.{args.source_lang}-{args.target_lang}.*")
#         else args.target_lang
#     )
#     second_l = (
#         args.target_lang
#         if glob.glob(f"{args.raw_data}/*.{args.source_lang}-{args.target_lang}.*")
#         else args.source_lang
#     )

#     train_inp_prefix = os.path.join(args.raw_data, f"train.tags.{first_l}-{second_l}")
#     train_out_prefix = os.path.join(
#         args.out_data, f"train.{args.source_lang}-{args.target_lang}"
#     )
#     extract_train(
#         f"{train_inp_prefix}.{args.source_lang}",
#         f"{train_out_prefix}.{args.source_lang}",
#         # docids_out=f"{train_out_prefix}.docids",
#     )
#     extract_train(
#         f"{train_inp_prefix}.{args.target_lang}",
#         f"{train_out_prefix}.{args.target_lang}",
#     )

#     # generate valid set based on devsets paseed
#     valid_out_prefix = os.path.join(
#         args.out_data, f"valid.{args.source_lang}-{args.target_lang}"
#     )
#     for devset in args.devsets:
#         valid_inp_prefix = os.path.join(
#             args.raw_data, f"IWSLT15.TED.{devset}.{first_l}-{second_l}"
#         )
#         extract_eval(
#             f"{valid_inp_prefix}.{args.source_lang}.xml",
#             f"{valid_out_prefix}.{args.source_lang}",
#             # docids_out=f"{valid_out_prefix}.docids",
#         )
#         extract_eval(
#             f"{valid_inp_prefix}.{args.target_lang}.xml",
#             f"{valid_out_prefix}.{args.target_lang}",
#         )

#     # generate valid set based on validsets paseed
#     valid_out_prefix = os.path.join(
#         args.out_data, f"valid.{args.source_lang}-{args.target_lang}"
#     )
#     for validset in args.validsets:
#         valid_inp_prefix = os.path.join(
#             args.raw_data, f"IWSLT15.TED.{validset}.{first_l}-{second_l}"
#         )
#         extract_eval(
#             f"{valid_inp_prefix}.{args.source_lang}.xml",
#             f"{valid_out_prefix}.{args.source_lang}",
#             # docids_out=f"{valid_out_prefix}.docids",
#         )
#         extract_eval(
#             f"{valid_inp_prefix}.{args.target_lang}.xml",
#             f"{valid_out_prefix}.{args.target_lang}",
#         )