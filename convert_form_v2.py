import sys




#Convert RWS's test data from "q1*q2<tab>q3*q4<tab>q5" to "q1<tab>q2<tab>1"
def convert_form_v2(inputPath,outputPath): #inpuPath="./testset2_v1_3.txt"
    '''
    :param inputPath: path of input file
    :param outputPath: path of output file
    :return index list of each line : [[0,10],...,[100,115]]
    '''
    with open(inputPath, mode='r', encoding='utf-8') as iF:
        new_file=[]
        index_list=[]
        begin=0
        end=0
        for line in iF:
            count=0
            line_list=line.strip().split('*')
            #print (df_test2.loc[i,0],df_test2.loc[i,1],df_test2.loc[i,2],df_test2.loc[i,3])
            for q2 in  line_list[1].strip().split('\t'):
                if q2!='':
                    count+=1
                    new_file.append('\t'.join([line_list[0].strip(),q2,'1']))
            for q3 in  line_list[2].strip().split('\t'):
                if q3!='':
                    count+=1
                    new_file.append('\t'.join([line_list[0].strip(),q3,'0']))
            for q4 in  line_list[3].strip().split('\t'):
                if q4!='':
                    count+=1
                    new_file.append('\t'.join([line_list[0].strip(),q4,'0']))

            end+=count
            index_list.append([begin,end])
            begin=end
    with open(outputPath, mode='w', encoding='utf-8') as oF:
        oF.write('\n'.join(new_file))
    return index_list


if __name__ == '__main__':
    convert_form_v2(sys.argv[1],sys.argv[2])
    print ('Done')

