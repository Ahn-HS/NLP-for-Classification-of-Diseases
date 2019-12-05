
def W2V_modeling(load_name, save_name):
    if os.path.isfile('data/modeling_result/' + save_name):
        print('"' + save_name + '" modeling file is exist already.\r\n')
        a = input('do you wanna modeling again? y or n\r\n')
        if a == 'y' or a == '':
            print('modeling start')
        else:
            print('기존 모델링 파일에서 유사어를 추출.\r\n')
            return

    data = word2vec.LineSentence('data/revision/' + load_name)
    model = word2vec.Word2Vec(data, size=100, window=10, min_count=4, hs=1, sg=1)  # 보통 차수는 50~100
    model.save("data/modeling_result/" + save_name)

    print("\r\nModeling is finished\r\n")


def W2V_similarity(load_file_name, keyword_name):
    print('==== "' + keyword_name + '" 키워드에 대한 유사어 추출 결과 ====')

    # Word2Vec 결과 메모리에 로드
    model = word2vec.Word2Vec.load("data/modeling_result/" + load_file_name)
    #Word2Vec 결과를 텍스트 형태로 저장 및 뷰
    # model.wv.save_word2vec_format('vector_to_text.txt')

    ############################### Word2Vec 유사단어 결과 보기 ####################
    try:
        similar_result = model.most_similar(positive=[keyword_name], topn=50)
        print(similar_result)

    except KeyError:
        print("존재하지 않음")
        return 0
    ################################

    # model.wv.most_similar_cosmul(positive=['지방', '억측'], negative=['돈'])
    # print(model.wv.doesnt_match("나는 우울하다".split()))

    # 유사 단어 관계 파악
    # model.wv.similarity('불안', '불면증')
    # print(model.score(["저는 너무 우울해요. 그리고 너무 억울해요 무가치 해요".split()]))

    del model
    return similar_result


# 모델링 결과 시각화 함수
def visualize():
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf", size = 3).get_name()
    rc('font', family=font_name)
    mpl.rcParams['axes.unicode_minus'] = False

    model_name = 'data/modeling_result/model_44t.model'
    model = g.Doc2Vec.load(model_name)

    vocab2 = ["불면", "불안", "초조", "신체증상", "무기력", "식욕부진", "우울감", "불안정성", "두통", "우울",
              "집중력감소", "자신감감소", "불안감", "피곤", "호소", "의욕감소", "신체", "극성", "속울렁",
              "쇄약", "침울", "흐느끼다", "허탈", "울적", "의욕상실", "기분들다", "쳐지다", "괜히", "시무룩", "짜증나다",
              "가식", "처량", "가라앉다", "감성적", "남", "겉모습", "증폭", "외롭다", "심란", "암담",
              "좌절", "하락", "목표", "부정적", "석사과정", "좌절감", "학업", "중도", "달성", "외교학",
              "공부", "일탈", "정치학", "경쟁심", "처량", "변리사", "실망", "불합격", "장학금"]

    vocab = []
    for keyword in model.wv.vocab:
        if keyword in vocab2:
            vocab.append(keyword)
    X = model[vocab]
    tsne = TSNE(n_components=2)

    # 100개의 단어에 대해서만 시각화
    X_tsne = tsne.fit_transform(X[:100, :])
    df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])

    print(df.head(1000))
    fig = plt.figure()
    fig.set_size_inches(500, 500)

    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])


    for word, pos in df.iterrows():
        # 침울에 대한 연관어 검정
        if word in ["불면", "불안", "초조", "신체증상", "무기력", "식욕부진", "우울감", "불안정성", "두통", "우울",
              "집중력감소", "자신감감소", "불안감", "피곤", "호소", "의욕감소", "신체", "극성", "속울렁",
              "쇄약"]:
            ax.annotate(word, pos, fontsize=15, color='black', fontweight='bold', )
            pass

        # 불면에 대한 연관어 파랑
        if word in ["침울", "흐느끼다", "허탈", "울적", "의욕상실", "기분들다", "쳐지다", "괜히", "시무룩", "짜증나다",
              "가식", "처량", "가라앉다", "감성적", "남", "겉모습", "증폭", "외롭다", "심란", "암담"]:
            ax.annotate(word, pos, fontsize=15, color='blue', fontweight='bold')
            pass

        # 좌절에 대한 연관어 녹색

        if word in ["좌절", "하락", "목표", "부정적", "석사과정", "좌절감", "학업", "중도", "달성", "외교학",
              "공부", "일탈", "정치학", "경쟁심", "처량", "변리사", "실망", "불합격", "장학금"]:
            ax.annotate(word, pos, fontsize=15, color='green', fontweight='bold')
            pass

    plt.title('"침울", "불면", "좌절" 키워드에 대하여 추출된 연관 키워드', fontsize=25)
    plt.show()


def all_keyword_model_similarity(Type, klist, file_name, save_exel_file_name):
    wb = openpyxl.Workbook()
    ws = wb.active
    font_bold = Font(bold=True)
    ws['A3'].font = font_bold
    ws['A3'] = 'keyword'
    ws['A5'].font = font_bold
    ws['A5'] = '유사어↓'
    row_count = 2

    ##################################### 모든 카테고리를 for문 키워드 추출 ##################################
    for index, keyword in enumerate(klist):
        index += index
        ws.cell(row=3, column=index+2, value=keyword).font = font_bold
        similar_list = W2V_similarity(file_name, keyword)
        if similar_list == 0:
            print('no similar word')
            continue
        else:
            pass

        for row, word in enumerate(similar_list):
            # 유사도 포함
            # ws.cell(row=row + 5, column=index + 2, value=str(word))
            ws.cell(row=row + 5, column=index + 2, value=word[0])
        row_count += 2
    ######################################

    if Type == 'depression':
        wb.save('data/similarity_result/depression_similar_result_depression_' + save_exel_file_name + '.xlsx')
    else:
        wb.save('data/similarity_result/similar_result_anxiety_' + save_exel_file_name + '.xlsx')



def all_execute(Type, revision_file, modelsave_file, klist, keysave_exelfile, load_file, save_file):
    W2V_modeling(revision_file, modelsave_file)
    all_keyword_model_similarity(Type, klist, modelsave_file, save_file)


def modeling_glove(version):
    r_ft = open('data/revision/revision_' + version + '.txt', 'r', encoding='utf-8')
    revision_line_list = []
    lines = r_ft.readlines()
    for idx, line in enumerate(lines):
        revision_line_list.append(line)

    vectorizer = CountVectorizer(min_df=10, ngram_range=(1, 1))
    X = vectorizer.fit_transform(revision_line_list)
    Xc = X.T * X  # co-occurrence matrix
    Xc.setdiag(0)  # 대각성분을 0으로
    result = Xc.toarray()  # array로 변환
    dic = {}
    for idx1, word1 in enumerate(result):
        tmpdic = {}
        for idx2, word2 in enumerate(word1):
            if word2 > 0:
                tmpdic[idx2] = word2
        dic[idx1] = tmpdic

    vocab = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
    vocab = [word[0] for word in vocab]

    model = glove.Glove(dic, d=100, alpha=0.75, x_max=100.0)
    for epoch in range(20):
        err = model.train(batch_size=200, workers=4)
        print("epoch %d, error %.3f" % (epoch, err), flush=True)

    # 단어벡터 추출
    wordvectors = model.W

    with open('glove_data/modeling_result/glove_modeling_result_' + version, 'wb') as f:
        pickle.dump([vocab, wordvectors], f)



# Glove 연관어를 추출하는 함수
def similarity_glove(version, keyword):
    with open('glove_data/modeling_result/glove_modeling_result_' + version, 'rb') as f:
        data = pickle.load(f)
    vocab = data[0]
    wordvectors = data[1]
    for index in most_similar(word=keyword, vocab=vocab, vecs=wordvectors, topn=50):
        print(str(index) + "\r\n")

    # print(most_similar(word='우울', vocab=vocab, vecs=wordvectors, topn=50))


# GloVe 결과에서 유사어 추출을 위한 함수
def most_similar(word, vocab, vecs, topn=10):
    query = vecs[vocab.index(word)]
    result = []
    for idx, vec in enumerate(vecs):
        if idx is not vocab.index(word):
            result.append((vocab[idx], 1 - cosine(query, vec)))
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return result[:topn]


def run_fasttext(version):
    with open('data/revision/revision_' + version + '.txt', 'r') as f:
        all_lines = f.readlines()
        revision_line_list = []
        revision_word_list = []
        for line in all_lines:
            for word in line.split(' '):
                revision_word_list.append(word.strip())
            revision_line_list.append(revision_word_list)
            revision_word_list = [None] * 0
            # print(revision_line_list)

    # # # 2차 배열을 통해서 [[한 진료 데이터의 단어들 각 1개씩], [한 진료 데이터의 단어들 각 1개씩]]
    # data = word2vec.LineSentence('data/revision/revision_' + version + '.txt')
    model = FastText(revision_line_list, min_count=3)
    # model.build_vocab(word_list)
    # model.train(word_list, total_examples=model.corpus_count, epochs=model.iter)
    similar_result = model.most_similar(positive=['침울'], topn=50)
    for i in similar_result:
        print(i)


# 중복되는 단어가 있는지 분석하는 함수
def overlap_word_matching(load_name):
    ################### 엑셀 파일 열기 #####################
    exel_file = openpyxl.load_workbook('data/similarity_result/similar_result_' + load_name + '.xlsx', data_only=True)
    excel_sheet = exel_file.get_sheet_by_name("Sheet")
    dic = {}

    for c_seq_1, col_1 in enumerate(excel_sheet.columns):
        if c_seq_1 > 0:
            if col_1[2].value is not None:
                print("=========================" + str(col_1[2].value) + "=========================")
                one_row_list = []
                for r_seq_1, row_1 in enumerate(excel_sheet.rows):
                    if r_seq_1 > 3:
                        one_row_list.append(str(row_1[c_seq_1].value))

                dic[str(col_1[2].value)] = one_row_list

    duplication_dic = {}
    for key, value_list in dic.items():
        if key is not None :
            for sequence in range(0, 50):
                if dic[key][sequence] != "None":
                    forward_value = dic[key][sequence]

                    duplication_list = []
                    for key2, value_list2 in dic.items():
                        for sequence2 in range(0, 50):
                            if dic[key][sequence2] != "None":
                                if forward_value == dic[key2][sequence2]:
                                    print(key + "with" + key2 + "   |   " + forward_value + " compare with " + dic[key2][sequence2])
                                    duplication_list.append(key2)
                    if len(duplication_list) > 1:
                        duplication_dic[forward_value] = duplication_list

    anal_sheet = exel_file.create_sheet('anlaysis')
    numeric = 0
    for KEY, VALUE in duplication_dic.items():
        anal_sheet.cell(row=numeric+2, column=2, value=KEY)
        anal_sheet.cell(row=numeric + 2, column=3, value=str(VALUE))
        numeric += 1
    exel_file.save('data/similarity_result/similar_result_' + load_name + '.xlsx')


def get_TFIDF(load_version):
    file = open('data/revision/revision_' + load_version + '.txt', 'r', encoding='utf-8-sig', newline='\n')
    file_text = file.read()
    remove_num_doc = re.sub(r'\d', '', file_text)
    remove_eng_doc = re.sub('[a-zA-Z]', '', remove_num_doc)
    doc_split = remove_eng_doc.split("\n")
    corpus_list = []
    for i, line in enumerate(doc_split):
        # 해당 문서에 2글자 이상인 것들만 사용
        if len(line.strip().split()) > 1:
            corpus_list.append(line.strip())

    ##################################################################
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus_list)

    word2id = defaultdict(lambda: 0)
    for idx, feature in enumerate(vectorizer.get_feature_names()):
        word2id[feature] = idx

    for i, sentence in enumerate(corpus_list):
        print('====== document[%d] ======' % i)
        print([(token, tfidf[i, word2id[token]]) for token in sentence.split()])
        if i > 1000:
            break
    ###################################################################

    # tfidf = TfidfVectorizer().fit_transform(corpus_list)
    # document_distances = (tfidf_matrix * tfidf_matrix.T)
    # print(document_distances.get_shape())
    # print(document_distances.toarray())


def get_Elmo_():
    elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)


def elmo_data_train(config):
    config = config
    # define ELMO model and dataset
    train_dataset = ElmoKoreanDataset(config)
    steps_per_epoch = int(train_dataset.get_corpus_size() / config["batch_size"])
    elmo = ELMO(config)

    # init train operations
    optimizer = tf.train.AdamOptimizer()
    train_global_steps = 0

    train_iter, train_batch = input_data(train_dataset)
    train_loss, train_acc, train_ops = \
        elmo.train(train_batch, train_global_steps)

    # init ELMO variables & session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    summary_merge = tf.summary.merge_all()
    writer = tf.summary.FileWriter(config["log_dir"] + '/train',
                                   sess.graph,
                                   filename_suffix=config["log_file_prefix"])


def get_XLNET():
    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training=True, is_finetune=True, FLAGS=FLAGS)
    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=input_ids,
        seg_ids=seg_ids,
        input_mask=input_mask)

    summary = xlnet_model.get_pooled_out(summary_type="last")

    seq_out = xlnet_model.get_sequence_output()


if __name__ == "__main__":
    start_time = time.time()

    e = int(time.time() - start_time)
    print('\r\n{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))