import sys
import collections
import pandas as pd

sys.path.append('..')
from model import Model

#set high and low values for 4 factor, 2 level, full factorial
a_low = 10**(-3)
a_high = 10**(-4)
b_low = 10**(-4)
b_high = 10**(-5)
c_low = 0.4
c_high = 0.8
d_low = 0.4
d_high = 0.8

#list the recipes per the levels, full factorial
null = [a_low, b_low, c_low, d_low]
a = [a_high, b_low, c_low, d_low]
b = [a_low, b_high, c_low, d_low]
ab = [a_high, b_high, c_low, d_low]
c = [a_low, b_low, c_high, d_low]
ac = [a_high, b_low, c_high, d_low]
bc = [a_low, b_high, c_high, d_low]
abc = [a_high, b_high, c_high, d_low]
d = [a_low, b_low, c_low, d_high]
ad = [a_high, b_low, c_low, d_high]
bd = [a_low, b_high, c_low, d_high]
abd = [a_high, b_high, c_low, d_high]
cd = [a_low, b_low, c_high, d_high]
acd = [a_high, b_low, c_high, d_high]
bcd = [a_low, b_high, c_high, d_high]
abcd = [a_high, b_high, c_high, d_high]


#flatten output list from each run {val_loss, val_acc, total_time} 
def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


# running 4 repications of a full factorial
recipe_list_rep1 = [null, a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd]
recipe_list_rep2 = [null, a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd]
recipe_list_rep3 = [null, a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd]
recipe_list_rep4 = [null, a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd]

## This handles the experiment
def experiment(recipe_list):
    all_results = []
    #iterate over
    for item in recipe_list:
        model_recipe = Model(test = True, learning_rate_1 = item[0], learning_rate_2 = item[1], dropout_rate_1 = item[2], dropout_rate_2 = item[3])
        one_result = model_recipe.runexample()
        all_results.append(flatten(one_result))
    return all_results

###
rep1_listout = generate_model(recipe_list_rep1)
rep2_listout = generate_model(recipe_list_rep2)
rep3_listout = generate_model(recipe_list_rep3)
rep4_listout = generate_model(recipe_list_rep4)


def csv_out(input_list):
    #make a data frame
    df = pd.DataFrame(input_lists)
    #output csv
    df.to_csv( str(input_list) + '_test2.csv', index=False)


    

    

    
# define which recipes you will run for each
recipe_list_rep1 = [null, a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd]
recipe_list_rep2 = [null, a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd]
recipe_list_rep3 = [null, a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd]
recipe_list_rep4 = [null, a, b, ab, c, ac, bc, abc, d, ad, bd, abd, cd, acd, bcd, abcd]

# define the list which will hold all data for each replication
rep1_listout = generate_model(recipe_list_rep1)
rep2_listout = generate_model(recipe_list_rep2)
rep3_listout = generate_model(recipe_list_rep3)
rep4_listout = generate_model(recipe_list_rep4)

#output each set of results to a file
csv_out(rep1_listout)
csv_out(rep2_listout)
csv_out(rep3_listout)
csv_out(rep4_listout)
