//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>
#pragma comment(lib,"pthreadVC2.lib")  

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_LINE_LRNGTH 10000
#define MAX_PARA 10

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING], wordclass_file[MAX_STRING], keyword_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING], read_wordvec_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int key = 0, epoch = 1;
int classes[MAX_PARA], tops[MAX_PARA];
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0;
real alpha = 0.025, speed = 1.0, starting_alpha, original_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;
long long iter_num = 200;

void InitUnigramTable() {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	i = 0;
	d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (real)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
/*
void ReadWord(char *word, FILE *fin) {
int a = 0, ch;
while (!feof(fin)) {
ch = fgetc(fin);
if (ch == 13) continue;
if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
if (a > 0) {
if (ch == '\n') ungetc(ch, fin);//ungetc()函数：把字符退回到输入流
break;
}
if (ch == '\n') {
//strcpy(word, (char *)"</s>");
strcpy_s(word, strlen("</s>")+1, (char *)"</s>");
return;
} else continue;
}
word[a] = ch;
a++;
if (a >= MAX_STRING - 1) a--;   // Truncate too long words
}
word[a] = 0;
}
*/


void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	while (!feof(fin)) {
		ch = fgetc(fin);
		if (ch == 13) continue;
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				break;
			}
			else continue; //prevent certain thread start to read sth like _word_
		}
		word[a] = ch;
		a++;
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING) length = MAX_STRING;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	//strcpy_s(vocab[vocab_size].word, length, word);
	vocab[vocab_size].cn = 0;
	vocab_size++;
	// Reallocate memory if needed
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	size = vocab_size;
	train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		// if (vocab[a].cn < min_count) {
		//  vocab_size--;
		//  free(vocab[vocab_size].word);
		//  } else {
		// Hash will be re-computed, as after the sorting it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
		if (vocab[a].cn == 0)
		{
			vocab[a].cn++;
		}
		train_words += vocab[a].cn;
		//}
	}
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
		vocab[b].cn = vocab[a].cn;
		vocab[b].word = vocab[a].word;
		b++;
	}
	else free(vocab[a].word);
	vocab_size = b;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			}
			else {
				min1i = pos2;
				pos2++;
			}
		}
		else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			}
			else {
				min2i = pos2;
				pos2++;
			}
		}
		else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}

void LearnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	fin = fopen(train_file, "rb");
	//fopen_s(&fin, train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	AddWordToVocab((char *)"</s>");
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		i = SearchVocab(word);
		if (i == -1) {
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		}
		else vocab[i].cn++;
		if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
	}
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	file_size = ftell(fin);
	fclose(fin);
}

void SaveVocab() {
	long long i;
	FILE *fo;
	fo = fopen(save_vocab_file, "wb");
	//fopen_s(&fo, save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

void ReadWordCountFromTrainfile(FILE *fin) {
	char word[MAX_STRING];
	long long a, i;
	train_words = 0;
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		train_words++;
		//printf("train_words_num=%ld\n", train_words);
		i = SearchVocab(word);
		if (i == -1) {
			a = SearchVocab((char *)"<unk>");
			vocab[a].cn++;
		}
		else vocab[i].cn++;
	}
}

void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin;
	fin = fopen(read_vocab_file, "rb");
	//fopen_s(&fin, read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	vocab_size = 0;
	AddWordToVocab((char *)"</s>");
	while (1) {
		ReadWord(word, fin);
		if (feof(fin)) break;
		i = SearchVocab(word);
		if (i == -1) {
			a = AddWordToVocab(word);
		}
		//if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
		//fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		//fscanf_s(fin, "%lld%c", &vocab[a].cn, strlen(&vocab[a].cn), &c, strlen(&c));
		//i++;
	}
	//add <unk> to map a word that doesn't in vocabulary
	i = SearchVocab((char *)"<unk>");
	if (i == -1) {
		a = AddWordToVocab((char *)"<unk>");
	}

	fin = fopen(train_file, "rb");
	//fopen_s(&fin, train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	ReadWordCountFromTrainfile(fin);

	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);

	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}

}



/*
void ReadVocab() {
long long a, i = 0;
char c;
char word[MAX_STRING];
//FILE *fin = fopen(read_vocab_file, "rb");
FILE *fin;
fopen_s(&fin,read_vocab_file, "rb");
if (fin == NULL) {
printf("Vocabulary file not found\n");
exit(1);
}
for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
vocab_size = 0;
while (1) {
ReadWord(word, fin);
if (feof(fin)) break;
a = AddWordToVocab(word);
//fscanf(fin, "%lld%c", &vocab[a].cn, &c);
fscanf_s(fin, "%lld%c", &vocab[a].cn, strlen(&vocab[a].cn), &c, strlen(&c));
i++;
}
SortVocab();
if (debug_mode > 0) {
printf("Vocab size: %lld\n", vocab_size);
printf("Words in train file: %lld\n", train_words);
}
//fin = fopen(train_file, "rb");
fopen_s(&fin,train_file, "rb");
if (fin == NULL) {
printf("ERROR: training data file not found!\n");
exit(1);
}
fseek(fin, 0, SEEK_END);
file_size = ftell(fin);
fclose(fin);
}
*/
void InitNet() {
	long long a, b;
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
	//syn0 = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);

	if (syn0 == NULL) { printf("Memory allocation failed\n"); exit(1); }
	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
		//syn1 = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
		if (syn1 == NULL) { printf("Memory allocation failed\n"); exit(1); }
		for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
			syn1[a * layer1_size + b] = 0;
	}
	if (negative>0) {
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
		//syn1neg = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
		if (syn1neg == NULL) { printf("Memory allocation failed\n"); exit(1); }
		for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
			syn1neg[a * layer1_size + b] = 0;
	}
	for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
		syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
	CreateBinaryTree();
}

void InitWordVec() {
	long long a;
	char word[MAX_STRING];
	char line[MAX_LINE_LRNGTH];
	int index;
	FILE *fin;
	fin = fopen(read_wordvec_file, "rb");
	//fopen_s(&fin, read_wordvec_file, "rb");
	while (1) {
		if (feof(fin)) break;
		ReadWord(word, fin);
		index = SearchVocab(word);
		if (index == -1) { 
			fgets(line, MAX_LINE_LRNGTH, fin); 
			continue;
		}//assume each float digits has least 10 char
		for (a = 0; a < layer1_size; a++) { 
			ReadWord(word, fin);
			syn0[index * layer1_size + a] = atof(word); 
		}
	}
	fclose(fin);
}

void *TrainModelThread(void *id) {
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long)id;
	real f, g;
	clock_t now;
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));//Xw
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));//e
															// FILE *fi = fopen(train_file, "rb");
	FILE *fi;
	fi = fopen(train_file, "rb");
	//fopen_s(&fi, train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	while (1) {
			if (word_count - last_word_count > 1000) {
				word_count_actual += word_count - last_word_count;
				last_word_count = word_count;
				if ((debug_mode > 1)) {
					now = clock();
					printf("%cAlpha: %.8f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
						word_count_actual / (real)(train_words + 1) * 100,
						word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
					fflush(stdout);
				}
				alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1) / speed);
				if (alpha < original_alpha * 0.0001) alpha = original_alpha * 0.0001;
			}
			if (sentence_length == 0) {
				while (1) {
					word = ReadWordIndex(fi);
					if (feof(fi)) break;
					if (word == -1) continue;
					word_count++;
					if (word == 0) break;
					// The subsampling randomly discards frequent words while keeping the ranking same
					if (sample > 0) {
						real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
						next_random = next_random * (unsigned long long)25214903917 + 11;
						if (ran < (next_random & 0xFFFF) / (real)65536) continue;
					}
					sen[sentence_length] = word;
					sentence_length++;
					if (sentence_length >= MAX_SENTENCE_LENGTH) break;
				}
				sentence_position = 0;
			}
			if (feof(fi)) break;
			if (word_count > train_words / num_threads) break;
			// prevent the first sentence cetain thread read is like <unk></s>
			//if (sentence_length == 0) continue;
			word = sen[sentence_position];
			if (word == -1) continue;
			for (c = 0; c < layer1_size; c++) neu1[c] = 0;
			for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
			next_random = next_random * (unsigned long long)25214903917 + 11;
			b = next_random % window;
			if (cbow) {  //train the cbow architecture
						 // in -> hidden
				for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
					c = sentence_position - window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
				}
				if (hs) for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * layer1_size;
					// Propagate hidden -> output
					for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * alpha;
					// Propagate errors output -> hidden
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
					// Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
				}
				// NEGATIVE SAMPLING
				if (negative > 0) for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					}
					else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = table[(next_random >> 16) % table_size];
						if (target == 0) target = next_random % (vocab_size - 1) + 1;
						if (target == word) continue;
						label = 0;
					}
					l2 = target * layer1_size;
					f = 0;
					for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
					if (f > MAX_EXP) g = (label - 1) * alpha;
					else if (f < -MAX_EXP) g = (label - 0) * alpha;
					else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
					for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
				}
				// hidden -> in
				for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
					c = sentence_position - window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
				}
			}
			else {  //train skip-gram
				for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
					c = sentence_position - window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					l1 = last_word * layer1_size;
					for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
					// HIERARCHICAL SOFTMAX
					if (hs) for (d = 0; d < vocab[word].codelen; d++) {
						f = 0;
						l2 = vocab[word].point[d] * layer1_size;
						// Propagate hidden -> output
						for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
						if (f <= -MAX_EXP) continue;
						else if (f >= MAX_EXP) continue;
						else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate
						g = (1 - vocab[word].code[d] - f) * alpha;
						// Propagate errors output -> hidden
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
						// Learn weights hidden -> output
						for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
					}
					// NEGATIVE SAMPLING
					if (negative > 0) for (d = 0; d < negative + 1; d++) {
						if (d == 0) {
							target = word;
							label = 1;
						}
						else {
							next_random = next_random * (unsigned long long)25214903917 + 11;
							target = table[(next_random >> 16) % table_size];
							if (target == 0) target = next_random % (vocab_size - 1) + 1;
							if (target == word) continue;
							label = 0;
						}
						l2 = target * layer1_size;
						f = 0;
						for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
						if (f > MAX_EXP) g = (label - 1) * alpha;
						else if (f < -MAX_EXP) g = (label - 0) * alpha;
						else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
						for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
					}
					// Learn weights input -> hidden
					for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
				}
			}
			sentence_position++;
			if (sentence_position >= sentence_length) {
				sentence_length = 0;
				continue;
			}
		}
	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

void TrainModel() {
	long a, b, c, d;
	FILE *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
	if (save_vocab_file[0] != 0) SaveVocab();
	if (output_file[0] == 0) return;
	InitNet();
	if (negative > 0) InitUnigramTable();
	if (read_wordvec_file[0] != 0) { printf("Initating word vectors from %s\n", read_wordvec_file); InitWordVec(); }
	start = clock();
	for (b = 0; b < epoch; b++) {
		printf("\nepoch %ld\n", b + 1);
		word_count_actual = 0;
		starting_alpha = alpha;
		for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
		for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	}
	fo = fopen(output_file, "wb");
	//fopen_s(&fo, output_file, "wb");
	// Save the word vectors
	printf("\nSave the word vectors\n");
	fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
	for (a = 0; a < vocab_size; a++) {
		fprintf(fo, "%s", vocab[a].word);
		if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
		else for (b = 0; b < layer1_size; b++) fprintf(fo, " %lf", syn0[a * layer1_size + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);

	if (key != 0) {
		// Run K-means on the word vectors
		if (wordclass_file[0] == 0 || keyword_file[0] == 0) { printf("NOT assign file name to record word classes\n"); return; }
		printf("Run K-means on the word vectors\n");
		FILE *fc, *fk;
		int i, clcn, top, closeid;
		int iter = iter_num;
		int *centcn, *cl;
		char tmp_file[MAX_STRING];
		char index[2];
		real closev, x;
		real *cent, *bestd;
		long long keyword_num, change;
		long long *bestw;
		//normalize
		real *norm;
		a = posix_memalign((void **)&norm, 128, (long long)vocab_size * layer1_size * sizeof(real));
		//norm = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
		if (norm == NULL) { printf("Memory allocation failed\n"); exit(1); }
		for (a = 0; a < vocab_size; a++) {
			closev = 0;
			for (b = 0; b < layer1_size; b++) closev += syn0[a*layer1_size + b] * syn0[a*layer1_size + b];
			closev = sqrt(closev);
			for (b = 0; b < layer1_size; b++) norm[a*layer1_size + b] = syn0[a*layer1_size + b] / closev;
		}
		for (i = 0; i < key; i++) {
			clcn = classes[i]; top = tops[i];
			keyword_num = 0, change = 0;
			centcn = (int *)malloc(clcn * sizeof(int));
			cl = (int *)calloc(vocab_size, sizeof(int));
			cent = (real *)calloc(clcn * layer1_size, sizeof(real));
			bestd = (real *)calloc(clcn * top, sizeof(real));
			bestw = (long long *)calloc(clcn * top, sizeof(long long));
			for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
			for (a = 0; a < iter; a++) {
				printf("%cIter %ld\t", 13, a + 1);
				fflush(stdout);
				for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
				for (b = 0; b < clcn; b++) centcn[b] = 1;
				for (c = 0; c < vocab_size; c++) {
					for (d = 0; d < layer1_size; d++) {
						cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
						centcn[cl[c]]++;
					}
				}
				for (b = 0; b < clcn; b++) {
					closev = 0;
					for (c = 0; c < layer1_size; c++) {
						cent[layer1_size * b + c] /= centcn[b];
						closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
					}
					closev = sqrt(closev);
					for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
				}
				change = 0;
				for (c = 0; c < vocab_size; c++) {
					closev = -10;
					closeid = 0;
					for (d = 0; d < clcn; d++) {
						x = 0;
						for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * norm[c * layer1_size + b];
						if (x > closev) {
							closev = x;
							closeid = d;
						}
					}
					if (cl[c] != closeid) change++;
					cl[c] = closeid;
				}
				printf("change %lld\n", change);
				fflush(stdout);
				if (change == 0) break;
			}

			printf("Save the K-means classes\n");
			index[0] = i + 48; index[1] = 0; 
			strcpy(tmp_file, wordclass_file); strcat(tmp_file, index);
			//strcpy_s(tmp_file, strlen(wordclass_file) + 1, wordclass_file); strcat_s(tmp_file, strlen(tmp_file) + 1, index);
			fc = fopen(tmp_file, "wb");
			//fopen_s(&fc, tmp_file, "wb");
			if (top) {
				for (a = 0; a < clcn * top; a++) bestd[a] = -10;
				for (a = 0; a < clcn * top; a++) bestw[a] = -1;
				for (a = 0; a < vocab_size; a++) {
					x = 0;
					for (b = 0; b < layer1_size; b++) x += cent[layer1_size * cl[a] + b] * syn0[a * layer1_size + b];
					for (b = 0; b < top; b++) {
						if (x > bestd[cl[a] * top + b]) {
							for (c = top - 1; c > b; c--) {
								bestd[cl[a] * top + c] = bestd[cl[a] * top + c - 1];
								bestw[cl[a] * top + c] = bestw[cl[a] * top + c - 1];
							}
							bestd[cl[a] * top + b] = x;
							bestw[cl[a] * top + b] = a;
							break;
						}
					}
				}
				for (a = 0; a < clcn * top; a++) {
					if (bestw[a] == -1) continue;
					fprintf(fc, "%s %d\n", vocab[bestw[a]].word, a / top);
					keyword_num++;
				}
				strcpy(tmp_file, keyword_file); strcat(tmp_file, index);
				//strcpy_s(tmp_file, strlen(keyword_file) + 1, keyword_file); strcat_s(tmp_file, strlen(tmp_file) + 1, index);
				fk = fopen(tmp_file, "wb");
				//fopen_s(&fk, tmp_file, "wb");
				fprintf(fk, "%lld %lld\n", keyword_num, layer1_size);
				for (a = 0; a < clcn * top; a++) {
					if (bestw[a] == -1) continue;
					fprintf(fk, "%s", vocab[bestw[a]].word);
					if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[bestw[a] * layer1_size + b], sizeof(real), 1, fk);
					else for (b = 0; b < layer1_size; b++) fprintf(fk, " %lf", syn0[bestw[a] * layer1_size + b]);
					fprintf(fk, "\n");
				}
				fclose(fk);
			}
			else for (a = 0; a < vocab_size; a++) fprintf(fc, "%s %d\n", vocab[a].word, cl[a]);
			free(bestw);
			free(bestd);
			free(centcn);
			free(cent);
			free(cl);
			fclose(fc);
		}
	}
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i, j;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-read-vec <file>\n");
		printf("\t\tThe inital word vectors will be read from <file>, not initiate randomly\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors\n");
		printf("\t-wordclass <file>\n");
		printf("\t\tUse <file> to save the resulting word clusters\n");
		printf("\t-keyword <file>\n");
		printf("\t\tUse <file> to save the resulting vectors of word clusters\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
		printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads, equal to epoch(default 1)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		printf("\t-cbow <int>\n");
		printf("\t\tUse the continuous back of words model; default is 0 (skip-gram model)\n");
		printf("\t-iter <int>\n");
		printf("\t\tSet max iteration number for clustering; default is 200\n");
		printf("\t-key <int>\n");
		printf("\t\tdo word cluster; default number of key is 0 (no word cluster step)\n");
		printf("\nExamples:\n");
		printf("./word2vec_mod -train train -output WordVectors.txt -read-vec newsblogbbs.txt -cbow 0 -size 200 -window 10 -negative 0 -hs 1 -sample 1e-3 -threads 25 -binary 0\n\n");
		return 0;
	}
	output_file[0] = 0;
	wordclass_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	read_wordvec_file[0] = 0;
	keyword_file[0] == 0;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	//if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy_s(train_file, strlen(argv[i + 1]) + 1, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	//if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy_s(save_vocab_file, strlen(argv[i + 1]) + 1, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	//if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy_s(read_vocab_file, strlen(argv[i + 1]) + 1, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vec", argc, argv)) > 0) strcpy(read_wordvec_file, argv[i + 1]);
	//if ((i = ArgPos((char *)"-read-vec", argc, argv)) > 0) strcpy_s(read_wordvec_file, strlen(argv[i + 1]) + 1, argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	//if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy_s(output_file, strlen(argv[i + 1]) + 1, argv[i + 1]);
	if ((i = ArgPos((char *)"-wordclass", argc, argv)) > 0) strcpy(wordclass_file, argv[i + 1]);
	//if ((i = ArgPos((char *)"-wordclass", argc, argv)) > 0) strcpy_s(wordclass_file, strlen(argv[i + 1]) + 1, argv[i + 1]);
	if ((i = ArgPos((char *)"-keyword", argc, argv)) > 0) strcpy(keyword_file, argv[i + 1]);
	//if ((i = ArgPos((char *)"-keyword", argc, argv)) > 0) strcpy_s(keyword_file, strlen(argv[i + 1]) + 1, argv[i + 1]);
	
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv))>0)iter_num = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-key", argc, argv)) > 0) key = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) 
		for (j = 0; j < key; j++) classes[j] = atoi(argv[i + 1 + j]);
	if ((i = ArgPos((char *)"-top", argc, argv)) > 0) 
		for (j = 0; j < key; j++) tops[j] = atoi(argv[i + 1 + j]);
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) epoch = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-speed", argc, argv)) > 0) speed = atoi(argv[i + 1]);

	original_alpha = alpha;
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	TrainModel();
	return 0;
}
