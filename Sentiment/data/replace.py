import sys
reload(sys)
sys.setdefaultencoding('utf8')

with open(sys.argv[1],'r') as filin, open(sys.argv[2],'w') as filout:
    line = filin.readline()
    word = []
    pos = []
    polo = []
    count = 0
    sentnum = 0
    lr = rr = 0
    while line:
        line = line.strip()
        print line.split(' ')[0]
        if len(line) != 0:
            buff1, buff2, buff3 = line.split('\t')
            if buff1 in ['-lrb-','-lqt-']:
                buff2='-LBR-'
                lr += 1
            if buff1 in ['-rrb-','-rqt-']:
                buff2='-RRB-'
                rr += 1
            word.append(buff1)
            pos.append(buff2)
            polo.append(buff3)
            count += 1
        if len(line) == 0:
            count += 1
            print 'Empty line at %d'%count
#            print ' '.join(word), '\n'
#            filout.write(' '.join(word) + '\n')
            for (wrd, pos, pol) in zip(word, pos, polo):
                filout.write(wrd+'\t'+ pos + '\t' + pol + '\n')
            filout.write('\n')
            word = []
            pos=[]
            polo = []
            sentnum += 1
        line = filin.readline()

    print('There should be %d sentence with %d words totally'%(sentnum, count))
    print('Totally replace %d -LRB- and  %d -RRB-' % (lr, rr))
