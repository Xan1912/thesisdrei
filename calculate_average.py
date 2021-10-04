'''Calculates average performance of 20 iterations'''

from statistics import mean

def parse(data):
    # class 0
    label0 = data[-5]
    label0 = label0.split('      ')
    pr_label0 = float(label0[2].replace(' ',''))
    re_label0 = float(label0[3])
    fs_label0 = float(label0[4])

    # class 1
    label1 = data[-4]
    label1 = label1.split('      ')
    pr_label1 = float(label1[2].replace(' ',''))
    re_label1 = float(label1[3])
    fs_label1 = float(label1[4])

    # macro avg
    mcr_avg = data[-2]
    mcr_avg = mcr_avg.split('      ')
    pr_mcr_avg = float(mcr_avg[1].replace(' ',''))
    re_mcr_avg = float(mcr_avg[2])
    fs_mcr_avg = float(mcr_avg[3])

    # weighted avg
    wgh_avg = data[-1]
    wgh_avg = wgh_avg.split('      ')
    pr_wgh_avg = float(wgh_avg[1].replace(' ',''))
    re_wgh_avg = float(wgh_avg[2])
    fs_wgh_avg = float(wgh_avg[3])

    result = {'pr_0':pr_label0,'re_0':re_label0,'fs_0':fs_label0,
                'pr_1':pr_label1,'re_1':re_label1,'fs_1':fs_label1,
                'mc_pr':pr_mcr_avg,'mc_re':re_mcr_avg,'mc_fs':fs_mcr_avg,
                'wg_pr':pr_wgh_avg,'wg_re':re_wgh_avg,'wg_fs':fs_wgh_avg}

    return result

# all log file paths
filepath_baseline_logs = 'Logs/Baseline_Logs/iter_{}.log'
filepath_baseline_polarity_logs = 'Logs/LSTMwithPolarityLogs/iter_{}.log'
filepath_BERT_naiveCSR_logs = 'Logs/BERTwithCSRnaiveLogs/iter_{}.log'
filepaths = [filepath_baseline_logs,filepath_baseline_polarity_logs,filepath_BERT_naiveCSR_logs]

for each in filepaths:
    for i in range(20):
        pr0=list();re0=list();fs0=list()
        pr1=list();re1=list();fs1=list()
        mcpr=list();mcre=list();mcfs=list()
        wgpr=list();wgre=list();wgfs=list()
        with open(each.format(i+1), 'r') as f:
            data = f.readlines()
            data = parse(data)
            pr_0 = data['pr_0']
            re_0 = data['re_0']
            fs_0 = data['fs_0']
            pr_1 = data['pr_1']
            re_1 = data['re_1']
            fs_1 = data['fs_1']
            mc_pr = data['mc_pr']
            mc_re = data['mc_re'] 
            mc_fs = data['mc_fs']
            wg_pr = data['wg_pr']
            wg_re = data['wg_re']
            wg_fs = data['wg_fs']

            pr0.append(pr_0)
            re0.append(re_0)
            fs0.append(fs_0)
            pr1.append(pr_1)
            re1.append(re_1)
            fs1.append(fs_1)
            mcpr.append(mc_pr)
            mcre.append(mc_re)
            mcfs.append(mc_fs)
            wgpr.append(wg_pr)
            wgre.append(wg_re)
            wgfs.append(wg_fs)

    final = {'pr_0':mean(pr0),'re_0':mean(re0),'fs_0':mean(fs0),
                'pr_1':mean(pr1),'re_1':mean(re1),'fs_1':mean(fs1),
                'mc_pr':mean(mcpr),'mc_re':mean(mcre),'mc_fs':mean(mcfs),
                'wg_pr':mean(wgpr),'wg_re':mean(wgre),'wg_fs':mean(wgfs)}

    print('Insert name of the result') # maybe we could make it as a dictionary
    print('     ','           Precision    ','   Recall    ', '   F-score    ')
    print ('0                   ', str(pr0[0]), '      ', '  ',str(re0[0]), '        ', str(fs0[0]), '      ')
    print ('1                   ', str(pr1[0]), '      ', ' ',str(re1[0]), '        ', str(fs1[0]), '      ')
    print ('Macro Average       ', str(mcpr[0]), '      ', ' ',str(mcre[0]), '        ', str(mcfs[0]), '      ')
    print ('Weighted Average    ', str(mcpr[0]), '      ', ' ',str(mcre[0]), '        ', str(mcfs[0]), '      ')
    print('\n')

