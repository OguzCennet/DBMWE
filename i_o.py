class Sentence:
	def __init__(self):
		self.form_list = []
		self.couple_list = []
		self.lemma_list = []
		self.pos_list = []
		self.int_seq = []
		self.arc_list = []
		self.morph_list = []
		
# ID FORM LEMMA UPOSTAG XPOSTAG FEATS HEAD DEPREL DEPS MISC		
# 0	 1	  2     3       4       5     6    7      8    9

class Dataset:
	def __init__(self, path):
		self.path = path
		self.sentences = []
		self.get_dataset()

		
	def get_dataset(self):
		lines = tuple(open(self.path, 'r'))
		sent_counter = 0
		sent = Sentence()
		for line in lines:
			#They used '-' char for turkish corpora instead of '.' 
			#if not (line.startswith('#') or '.' in line.split("\t")[0]):
			if not (line.startswith('#') or '.'  in line.split("\t")[0] or '-' in line.split("\t")[0]):
				if line.strip():
					attr = line.split("\t")
					if int(attr[0]) == 1:
						sent.arc_list.append((-1,0))
						sent.int_seq.append(0)
						sent.form_list.append("ROOT")
						sent.lemma_list.append("ROOT")
						sent.pos_list.append("ROOT")
						sent.morph_list.append("ROOT")
						if attr[6] == '_':
							sent.arc_list.append((attr[6],attr[0]))
							sent.couple_list.append((attr[0],attr[1] ))
						else:
							sent.arc_list.append((int(attr[6]),int(attr[0])))
							sent.couple_list.append((attr[0],attr[1] ))
						sent.pos_list.append(attr[3])
						sent.int_seq.append(int(attr[0]))
						sent.form_list.append(attr[1])
						sent.morph_list.append(attr[5])
						sent.lemma_list.append(attr[2])
					else:
						if attr[6] == '_':
							sent.arc_list.append((attr[6],attr[0]))
							sent.couple_list.append((attr[0],attr[1] ))

						else:
							sent.arc_list.append((int(attr[6]),int(attr[0])))
							sent.couple_list.append((attr[0],attr[1] ))
						sent.pos_list.append(attr[3])
						sent.int_seq.append(int(attr[0]))
						sent.form_list.append(attr[1])
						sent.lemma_list.append(attr[2])
						sent.morph_list.append(attr[5])
				else:
					self.sentences.append(sent)
					sent = Sentence()
					sent_counter +=1
				
			if sent_counter  == 100:
				break
		


	
	

		
		
