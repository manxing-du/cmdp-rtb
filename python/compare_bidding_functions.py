import pandas as pd
import glob
import numpy as np


bid_log_path = '../log/bid-log/'

ipinyou_camps = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]
bid_functions = ['lin','mcpc','rlb-compete']
bid_functions_CMDP = ['CMDP','batch-CMDP']

c0 = 1/32
src = 'ipinyou'

setting = "compare_bids_{}, c0={}" .format(src, c0)
log_path = "{}.txt".format(setting)
outfile = open(log_path, 'w')


def find_secondprice(row):
    try:
        row['second_price'] = sorted(set(row))[-2]
    except IndexError:
        row['win_function'] = 'win'
        row['second_price'] = 0
    return row['second_price']


for camp in ipinyou_camps:
    df_bid_price = pd.DataFrame()
    # Read mcpc, rlb, lin bid logs
    for function in bid_functions:
        files = glob.glob(bid_log_path + function + '/' + src + ', ' + 'camp=' + camp + '*c0=' + str(c0) +'*.txt')

        log_in = pd.read_csv(files[0],
                             header=None, index_col=False, sep='\t',
                             names=['time', 'episode', 'budget_action', 'bid_win_click', 'clk_imp']
                             )

        df_bid_price[function] = log_in.apply(lambda row: row['bid_win_click'].split('_')[0], axis=1)


    # Read batch-CMDP
    for function in bid_functions_CMDP:

        files = glob.glob(bid_log_path + function + '/' + src + '_camp=' + camp + '*c0=' + str(c0) +'*.txt')
        log_in = pd.read_csv(files[0], index_col=False, sep='\t', header=0, skipinitialspace=True, dtype=int)
        log_in.columns = [function, 'win', 'click', 'budget']
        df_bid_price[function] = log_in[function]
        df_bid_price['win'] = log_in['win']
        df_bid_price['click'] = log_in['click']

    ctr = float(df_bid_price[df_bid_price.click == 1].shape[0]) / float(df_bid_price.shape[0]) * 100
    outfile.write("{:<10} {:<2}\t{:<10} {:<8.4f}%\n".format('camp', camp, 'CTR', ctr))
    print('camp ' + str(camp))
    print('CTR ' + str(ctr))
    print('total auctions ' + str(df_bid_price.shape[0]))
    # Compare bid price
    print('start comparing')
    functions = ['lin', 'mcpc', 'rlb-compete', 'CMDP', 'batch-CMDP', 'win']

    df_bid_price[['lin', 'mcpc', 'rlb-compete', 'CMDP', 'batch-CMDP', 'win']] \
        = df_bid_price[['lin', 'mcpc', 'rlb-compete', 'CMDP', 'batch-CMDP', 'win']].fillna(0).astype(int)
    df_bid_price['win_function'] = df_bid_price[['lin','mcpc','rlb-compete', 'CMDP', 'batch-CMDP', 'win']].apply(
        lambda row: np.argwhere(row == np.amax(row)).flatten().tolist(), axis=1)

    df_bid_price["second_price"] = df_bid_price[['lin', 'mcpc', 'rlb-compete', 'CMDP', 'batch-CMDP', 'win']]\
        .apply(find_secondprice, axis=1)

    multi_win = 0

    multi_win_func = {}
    single_win = {}
    total_clk = df_bid_price[df_bid_price.click == 1].shape[0]
    print("total clicks is " + str(total_clk))

    # How many of them have multiple wins
    for idx, row in df_bid_price.iterrows():
        if (len(row['win_function']) > 1) & (5 not in row['win_function']):
            multi_win += 1
            for i in row['win_function']:
                if functions[i] not in multi_win_func:
                    multi_win_func[functions[i]] = {}
                    multi_win_func[functions[i]]['multi_imp'] = 1
                    multi_win_func[functions[i]]['multi_clk'] = row['click']
                    multi_win_func[functions[i]]['multi_cost'] = row['second_price']
                else:
                    multi_win_func[functions[i]]['multi_imp'] += 1
                    multi_win_func[functions[i]]['multi_clk'] += row['click']
                    multi_win_func[functions[i]]['multi_cost'] += row['second_price']

        elif (len(row['win_function']) > 1) & (5 in row['win_function']):
            if 'win' not in single_win:
                single_win['win'] = {}
                single_win['win']['single_imp'] = 1
                single_win['win']['single_clk'] = row['click']
                single_win['win']['single_cost'] = 0
            else:
                single_win['win']['single_imp'] += 1
                single_win['win']['single_clk'] += row['click']
                single_win['win']['single_cost'] += 0

        elif len(row['win_function']) == 1:
            for i in row['win_function']:
                if functions[i] not in single_win:
                    single_win[functions[i]] = {}
                    single_win[functions[i]]['single_imp'] = 1
                    single_win[functions[i]]['single_clk'] = row['click']
                    single_win[functions[i]]['single_cost'] = row['second_price']
                else:
                    single_win[functions[i]]['single_imp'] += 1
                    single_win[functions[i]]['single_clk'] += row['click']
                    single_win[functions[i]]['single_cost'] += row['second_price']
    print("Total multi_winning_imp is " + str(multi_win))

    for function in functions:
        try:
            single_win[function]['single_ecpc'] = float(single_win[function]['single_cost']) \
                                                  / float(single_win[function]['single_clk'])
        except KeyError:
            pass
        except ZeroDivisionError:
            single_win[function]['single_ecpc'] = 0

        try:
            multi_win_func[function]['multi_ecpc'] = float(multi_win_func[function]['multi_cost']) \
                                                     / float(multi_win_func[function]['multi_clk'])
        except KeyError:
            pass
        except ZeroDivisionError:
            multi_win_func[function]['multi_ecpc'] = 0

    M_win = pd.DataFrame.from_records(multi_win_func)
    S_win = pd.DataFrame.from_records(single_win)

    print(M_win)
    print(S_win)

    total_cpc = []
    results = pd.concat([M_win,S_win])

    for column in results:
        total_cpc.append((results[column]['multi_cost'] + results[column]['single_cost'])
                         / (results[column]['multi_clk'] + results[column]['single_clk']))

    results.loc['total_ecpc'] = total_cpc
    results.fillna(0,inplace=True)
    for i in ['CMDP', 'batch-CMDP', 'lin', 'mcpc', 'rlb-compete', 'win']:
        if i not in results.columns:
            results[i] = np.zeros(results.shape[0])

    print(results)
    log = "{:>20}\t{:>20}\t{:>20}\t{:>20}\t{:>20}\t{:>20}\t{:>20}\n".format(' ', 'CMDP', 'batch-CMDP', 'lin', 'mcpc',
                                                                    'rlb-compete', 'win')
    outfile.write(log)
    for index, row in results.iterrows():
        log = "{:>20}\t{:>20}\t{:>20}\t{:>20}\t{:>20}\t{:>20}\t{:>20}\n"\
            .format(index, row['CMDP'], row['batch-CMDP'], row['lin'], row['mcpc'], row['rlb-compete'], row['win'])
        outfile.write(log)

outfile.close()

