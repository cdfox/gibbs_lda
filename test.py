import lda

docs = ['cat cat', 'cat dog']

print 'docs:', docs
print

tokenized_docs = lda.tokenize(docs, min_times=1, max_ratio=1.0, 
							  min_word_size=1)

print 'tokenized docs:', tokenized_docs
print

lda = lda.LDASampler(
	docs=tokenized_docs, 
	num_topics=2, 
	alpha=0.25,
	beta=0.25)

print 'topic assignments for each of 10 iterations of sampling:'
for _ in range(10):
	zs = lda.assignments
	print '[%i %i] [%i %i]' % (zs[0][3], zs[1][3], zs[2][3], zs[3][3])
	lda.next()
print

print 'words ordered by probability for each topic:'
tks = lda.topic_keys()
for i, tk in enumerate(tks):
	print i, tk
print

print 'document keys:'
dks = lda.doc_keys()
for doc, dk in zip(docs, dks):
	print doc, dk
print

print 'topic assigned to each word of first document in the final iteration:'
lda.doc_detail(0)
