# =============================================================================
# 
# =============================================================================

import sys
import jieba
import jieba.posseg
# import imp


if __name__ == "__main__":
    # imp.reload(sys)
    # sys.setdefaultencoding('utf-8')
    f = open('.\\novel.txt', encoding='utf-8')
    str = f.read()#.decode('utf-8')
    f.close()

    seg = jieba.posseg.cut(str)
    for word, pos in seg:
        # print(word, '|', end=' ')
        print(word, pos, '|', end=' ')