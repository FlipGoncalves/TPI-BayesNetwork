#encoding: utf8
# Student: Filipe Gonçalves, 98083

# Discussion of the work with students:
# Pedro Lopes, 97827
# João Borges, 98155
# Gonçalo Machado, 98359
# Vicente Costa, 98515
# Catarina Oliveira, 98292


from semantic_network import *
from bayes_net import *
from functools import reduce


class MySemNet(SemanticNetwork):
    def __init__(self):
        SemanticNetwork.__init__(self)
        self.conf = lambda n,T: n/ (2*T) + (1 - n / (2*T)) * (1 - (0.95 ** n)) * (0.95 ** (T - n))

    def source_confidence(self,user):
        correct = 0
        wrong = 0

        common_map = {}
        for d in self.declarations:
            ent = d.relation.entity1
            if isinstance(d.relation, AssocOne):
                if d.relation.name in common_map.keys():
                    if ent in common_map[d.relation.name]:
                        common_map[d.relation.name][ent].append(d.relation.entity2)
                    else:
                        common_map[d.relation.name][ent] = [d.relation.entity2]
                else:
                    common_map[d.relation.name] = {ent:[d.relation.entity2]}

        for ind in common_map:
            for indx in common_map[ind]:
                common_map[ind][indx] = {i:common_map[ind][indx].count(i) for i in common_map[ind][indx]}

        new_dic = {}
        for ind in common_map:
            for indx in common_map[ind]:
                max_ind = max(common_map[ind][indx], key=common_map[ind][indx].get)
                for tupple in common_map[ind][indx]:
                    if common_map[ind][indx][tupple] == common_map[ind][indx][max_ind]:
                    # if tupple == max_ind:
                        if ind in new_dic.keys():
                            if indx in new_dic[ind]:
                                new_dic[ind][indx].append(tupple)
                            else:
                                new_dic[ind][indx] = [tupple]
                        else:
                            new_dic[ind] = {indx:[tupple]}
              
        for d in self.declarations:
            if isinstance(d.relation, AssocOne) and d.user == user:
                if d.relation.entity2 in new_dic[d.relation.name][d.relation.entity1]:
                    correct += 1
                else:
                    wrong += 1

        return (1-(0.75**correct))*(0.75**wrong)

    def query_with_confidence(self,entity,assoc):
        local = [c for c in self.query_local(e1=entity, relname=assoc) if isinstance(c.relation, AssocOne)]

        count_map = {}
        for c in local:
            e2 = c.relation.entity2
            if e2 in count_map:
                count_map[e2] += 1
            else:
                count_map[e2] = 1

        comp = {}
        count_map_rec = {}
        pds = [c for c in self.query_local(e1=entity) if isinstance(c.relation, Subtype) or isinstance(c.relation, Member)]
        for c in pds:
            comp = self.query_with_confidence(c.relation.entity2, assoc)
            for key, value in comp.items():
                if key in count_map_rec:
                    count_map_rec[key] += value
                else:
                    count_map_rec[key] = value

        if len(local) == 0:
            for c in count_map_rec:
                count_map[c] = (count_map_rec[c] / len(pds)) * 0.9
        elif len(comp) == 0:
            for c in count_map:
                count_map[c] = self.conf(count_map[c], len(local)) 
        else:
            keylist = list(set(list(count_map_rec.keys()) + list(count_map.keys())))
            for c in keylist:
                if c in count_map and c in count_map_rec:
                    count_map[c] = self.conf(count_map[c], len(local)) * 0.9 + count_map_rec[c] / (len(pds)) * 0.1
                elif c in count_map_rec: 
                    count_map[c] = count_map_rec[c] / (len(pds)) * 0.1
                else: 
                    count_map[c] = self.conf(count_map[c], len(local)) * 0.9
        
        return count_map


class MyBN(BayesNet):

    def __init__(self):
        BayesNet.__init__(self)
        # IMPLEMENT HERE (if needed)
        pass

    def individual_probabilities(self):
        variables = [k for k in self.dependencies.keys()]

        my_dic = {}
        for var in variables:
            result = 0
            con = {}
            for key in self.dependencies[var].keys():
                con[key] = self.dependencies[var][key]

            for c in con:
                temp = list(c)
                temp_2 = []
                for t in temp:
                    if t[1]:
                        temp_2.append(my_dic[t[0]])
                    else:
                        temp_2.append(1 - my_dic[t[0]])
                if temp_2:
                    result += con[c] * reduce((lambda x,y: x * y), temp_2)

            if result == 0:
                for key in self.dependencies[var].keys():
                    result += self.dependencies[var][key]

            my_dic[var] = result

        return my_dic
        