# -*- coding: utf-8 -*-

# Since we have the types and allocations we can make lots of cool experiments
from collections import defaultdict
import click
import math
import itertools
from scipy.stats.stats import pearsonr

import pandas as pd
import numpy as np
from os import listdir, path
import matplotlib

matplotlib.use('Agg')


#### Dummy to replace seaborn
class MayBeCalled(object):
    def __call__(self, *args, **kwargs):
        return None

class Dummy(object):
    def __getattr__(self, attr):
        return MayBeCalled()

    def __setattr__(self, attr, val):
        pass

sns = Dummy()

# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from file_name import FileName

sns.set_palette(sns.color_palette("Paired", 10))


i18n = {'memory': u"Memória", 'cpu': "CPU"}

sns.set_style('ticks')


class Allocation(object):
    def __init__(self, data_path):
        files = sorted(listdir(data_path))
        files = filter(lambda x: x[-4:] == '.csv', files)
        alloc_files = filter(lambda x: x.startswith('alloc'), files)
        alloc_files = map(FileName, alloc_files)
        alloc_files.sort(key=lambda f: f.resource_percentage)

        user_needs_files = filter(lambda x: x.startswith('user_needs'), files)
        user_needs_files = map(FileName, user_needs_files)
        user_needs_files = {n.attributes['resource_type']: path.join(
            data_path, n.name) for n in user_needs_files}
        self._allocations = defaultdict(list)
        for f in alloc_files:
            params_dict = f.attributes
            resource_type = params_dict['resource_type']
            params_dict['file_name'] = path.join(data_path, f.name)
            params_dict['types_file_name'] = user_needs_files[resource_type]
            self._allocations[resource_type].append(params_dict)
        self.user_type_files = user_needs_files

    def resource_types(self):
        return self._allocations.keys()

    def __getattr__(self, name):
        return self._allocations[name]

    def iteritems(self):
        return self._allocations.iteritems()


# r_i vs Parcela de necessidades atendidas para vários gamas
def request_fulfilment(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        attended_requests_by_delta = defaultdict(list)
        file_name = alloc.user_type_files[resource_type]
        user_needs = pd.read_csv(file_name, index_col='time')
        num_reqs = (user_needs > 0).sum().sum()
        res_means = user_needs.mean()

        user_types_cache = {}

        def get_user_types(resource_percentage):
            if resource_percentage in user_types_cache:
                return user_types_cache[resource_percentage]
            user_resources = resource_percentage * res_means
            user_resources = user_resources.apply(
                lambda x: np.ceil(x).astype(np.int64))
            user_resources[user_resources.index.str.contains('fr')] = 0
            user_types = user_needs - user_resources
            user_types_cache[resource_percentage] = user_types

            attended_requests = (user_types[user_needs > 0] <= 0
                                 ).sum().sum() * 100.0 / num_reqs
            data_point = (resource_percentage, attended_requests)
            attended_requests_by_delta[10.0].append(data_point)
            return user_types_cache[resource_percentage]

        marker = itertools.cycle(('+', '.', 'o', '*', '^', 's','d'))

        for a in allocations:
            allocation = pd.read_csv(a['file_name'], index_col='time')
            resource_percentage = a['resource_percentage']
            user_types = get_user_types(resource_percentage)

            attended_requests = (allocation[user_needs > 0] >=
                                 user_types[user_needs > 0]
                                 ).sum().sum() * 100.0 / num_reqs

            data_point = (resource_percentage, attended_requests)
            attended_requests_by_delta[a['delta']].append(data_point)

        f, ax = plt.subplots(figsize=(6, 3))
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))

        for delta in sorted(attended_requests_by_delta.iterkeys()):
            print delta
            points = attended_requests_by_delta[delta]
            points.sort()
            print zip(*points)
            if delta == 10.0:
                label = u'Sem mecanismo'
            else:
                label = "$\delta = %.5f$" % delta
            plt.plot(*zip(*points), label=label, marker=marker.next())
        # plt.title(u"Necessidades de %s Satisfeitas" % i18n[resource_type])
        plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
        plt.ylabel(u"Requisições Satisfeitas (%)")
        plt.legend(loc=4)
        sns.despine()
        plt.savefig('fulfilment-%s.pdf' % resource_type, bbox_inches='tight')
        plt.close()


# r_i vs u_i médio para vários deltas
def resource_vs_utility(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        average_utility_by_delta = defaultdict(list)

        for a in allocations:
            allocation = pd.read_csv(a['file_name'], index_col='time')
            # As our allocation works under the mechanism constraints we can
            # infer that any positive allocation will also be the utility.
            # Also, we can infer that any negative allocation will translate
            # into a zero utility
            utility = allocation.copy()
            utility[utility < 0] = 0
            average_utility = utility.mean().mean()
            data_point = (a['resource_percentage'], average_utility)
            average_utility_by_delta[a['delta']].append(data_point)
        f, ax = plt.subplots(figsize=(6, 4))
        for delta in sorted(average_utility_by_delta.iterkeys()):
            print delta
            points = average_utility_by_delta[delta]
            points.sort()
            print zip(*points)
            plt.plot(*zip(*points), label="$\delta = %.5f$" % delta)
        # plt.title(u"Média das Utilidades para %s" % i18n[resource_type])
        plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
        plt.ylabel(u"Média das Utilidades dos Usuários")
        plt.legend(loc=4)
        sns.despine()
        plt.savefig('mean_utility-%s.pdf' % resource_type, bbox_inches='tight')
        plt.close()


# Histograma com as credibilidades dos usuários no fim das simulações para cada
# gama e para cada r_i Parcela
def credibility_distribution(data_path):
    files = sorted(listdir(data_path))
    credibility_files = filter(lambda x: x.startswith('credibility'), files)
    credibility_files = map(FileName, credibility_files)
    data_by_res_percentage_and_type = defaultdict(dict)

    for f in credibility_files:
        delta = f.attributes['delta']
        resource_percentage = f.attributes['resource_percentage']
        resource_type = f.attributes['resource_type']
        x = np.genfromtxt(path.join(data_path, f.name), delimiter=',')
        data_by_res_percentage_and_type[
            (resource_percentage, resource_type)][delta] = x
    df_list_per_res_type = defaultdict(list)
    for (resource_percentage, resource_type), data in \
            data_by_res_percentage_and_type.iteritems():
        df = pd.DataFrame(data)
        df = pd.melt(df)
        df['resource_percentage'] = resource_percentage
        df_list_per_res_type[resource_type].append(df)

    df_per_res_type = {}
    for k, v in df_list_per_res_type.iteritems():
        df_per_res_type[k] = pd.concat(v)

    for resource_type, df in df_per_res_type.iteritems():
        f, ax = plt.subplots()
        ax.set(yscale="symlog")

        ax = sns.swarmplot(x='variable', y='value', hue='resource_percentage',
                           data=df, palette='GnBu_d', split=True, size=2)
        plt.title(u"Distribuição de Credibilidade para %s" %
                  i18n[resource_type])
        plt.xlabel(u"$\delta$")
        plt.ylabel(u"Credibilidade no Fim das Alocações")
        plt.legend(title=u'Recursos', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()
        plt.savefig('credibility_distribution-%s.pdf' % resource_type,
                    bbox_inches='tight')
        plt.close()


# Histograma com a soma de recursos utilizados comparando os casos com e sem o
# mecanismo (pode ser interessante também comparar com o caso em que todas as
# requisições são satisfeitas)
def resource_utilization_bar_plot(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        file_name = alloc.user_type_files[resource_type]
        user_needs = pd.read_csv(file_name, index_col='time')
        res_means = user_needs.mean()

        # plt.rc('xtick', labelsize=15)
        # plt.rc('ytick', labelsize=15)
        # plt.rcParams.update({'font.size': 18})

        f, ax = plt.subplots(figsize=(6, 3))
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)

        percentages = []
        labels = []

        for a in allocations:
            if a['delta'] != 0.0:
                continue
            allocation = pd.read_csv(a['file_name'], index_col='time')
            resource_percentage = a['resource_percentage']
            user_resources = resource_percentage * res_means
            user_resources = user_resources.apply(
                lambda x: np.ceil(x).astype(np.int64))
            user_resources[user_resources.index.str.contains('fr')] = 0
            available_resources = user_resources + allocation
            used_resources = user_needs.where(user_needs < available_resources,
                                              available_resources)
            used_resources_sum = used_resources.sum(axis=1).as_matrix()
            print used_resources_sum

            num_elements = len(used_resources_sum)
            res_available = sum(user_resources)
            complete_usage_count = list(used_resources_sum).count(res_available)
            percentages.append(complete_usage_count*100.0/num_elements)
            labels.append(resource_percentage)

        ax.bar(range(1, len(percentages) + 1), percentages, tick_label=labels)

        plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
        plt.ylabel(u"Iterações (%)")
        # plt.legend(title=u'Recursos', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sns.despine()
        plt.savefig('resource_utilization-%s.pdf' % resource_type,
                    bbox_inches='tight', dpi=1200)
        plt.close()


# Recursos recebidos vs recursos contribuídos para diferentes deltas e r_i
# Ideia: Calcular correlação entre recursos recebidos e contribuídos, gráfico
# pode ter:
# eixo x: Percentual de Recursos
# eixo y: Correlação calculada
# cores: deltas
def resources_received_vs_given(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        df_list = []

        corr_by_delta = defaultdict(list)

        for a in allocations:
            resource_percentage = a['resource_percentage']
            delta = a['delta']
            allocation = pd.read_csv(a['file_name'], index_col='time')
            received_resources = allocation[allocation >= 0].sum()
            contributed_resources = -1 * allocation[allocation <= 0].sum()
            received_resources = received_resources[contributed_resources > 0]
            contributed_resources = \
                contributed_resources[contributed_resources > 0]
            contributed_resources = contributed_resources[received_resources > 0]
            received_resources = received_resources[received_resources > 0]
            corr = pearsonr(contributed_resources, received_resources)[0]
            corr_by_delta[delta].append((resource_percentage, corr))

        f, ax = plt.subplots(figsize=(6, 3))
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        marker = itertools.cycle(('+', '.', 'o', '*', '^', 's','d'))
        for delta in sorted(corr_by_delta.iterkeys()):
            print delta
            points = corr_by_delta[delta]
            points.sort()
            label = "$\delta = %.5f$" % delta
            plt.plot(*zip(*points), label=label, marker=marker.next())

        plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
        plt.ylabel(u"Correlação de Pearson")
        plt.ylim([0.0, 1])
        plt.legend(ncol=2)
        plt.savefig('resources_received_vs_given-%s.pdf' %
                (resource_type), bbox_inches='tight')
        plt.close()


# r_i vs razão de recursos fornecidos e recebidos para vários gamas
def request_justice(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        attended_requests_by_delta = defaultdict(list)

        for a in allocations:
            allocation = pd.read_csv(a['file_name'], index_col='time')
            users_justice = - allocation[allocation > 0].sum() / \
                              allocation[allocation < 0].sum()
            # users_justice_mean = users_justice.mean()
            data_point = (a['resource_percentage'], users_justice)
            attended_requests_by_delta[a['delta']].append(data_point)
        for delta in sorted(attended_requests_by_delta.iterkeys()):
            print delta
            points = attended_requests_by_delta[delta]
            f, ax = plt.subplots(figsize=(8, 8))
            ax.set(yscale="log")
            df = pd.DataFrame({k: v for k, v in points})
            ax = sns.swarmplot(data=df,  ax=ax)

            # plt.title(u"Relação Entre a Média de %s Recebida e Contribuída"
            #          % i18n[resource_type])
            plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
            plt.ylabel(u"Relação entre %s" % i18n[resource_type])
            plt.legend()
            sns.despine()
            plt.savefig('justice-%s-%.5f.pdf' % (resource_type, delta),
                        bbox_inches='tight')
            plt.close()


# Quantidade de recursos atribuídos a um usuário em um conflito vs soma dos
# tipos anteriores deste usuário
# Ideia: Analisar quanto de recurso faltou em cada requisição e comparar com a
# soma dos tipos passados
def conflict_resources(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        print resource_type
        attended_requests_by_delta = defaultdict(list)
        file_name = alloc.user_type_files[resource_type]
        user_needs = pd.read_csv(file_name, index_col='time')
        num_reqs = (user_needs > 0).sum().sum()
        res_means = user_needs.mean()

        user_types_cache = {}

        def get_user_types(resource_percentage):
            if resource_percentage in user_types_cache:
                return user_types_cache[resource_percentage]
            user_resources = resource_percentage * res_means
            user_resources = user_resources.apply(
                lambda x: np.ceil(x).astype(np.int64))
            user_resources[user_resources.index.str.contains('fr')] = 0
            user_types = user_needs - user_resources
            user_types_cache[resource_percentage] = user_types

            attended_requests = (user_types[user_needs > 0] <= 0
                                 ).sum().sum() * 1.0 / num_reqs
            data_point = (resource_percentage, attended_requests)
            attended_requests_by_delta[-1.0].append(data_point)
            return user_types_cache[resource_percentage]

        data_by_res_percent = defaultdict(dict)

        for a in allocations:
            print 'delta: ', a['delta']
            print 'rp: ', a['resource_percentage']
            allocation = pd.read_csv(a['file_name'], index_col='time')
            resource_percentage = a['resource_percentage']
            user_types = get_user_types(resource_percentage)
            deficit = user_types - allocation

            # select rows where there is conflict
            conflict_rows = (deficit > 0).any(axis=1)
            conflict_deficit = deficit[conflict_rows]
            conflict_deficit = conflict_deficit[user_types > 0]

            # failed_requests = deficit[deficit > 0].as_matrix().flatten()
            allocation_mean = pd.expanding_mean(allocation).shift() #  allocation.cumsum().shift()
            allocation_mean.iloc[0] = 0
            allocation_mean = allocation_mean[conflict_rows][user_types > 0].as_matrix().flatten()

            conflict_deficit = conflict_deficit.as_matrix().flatten()

            # sns.distplot(type_sum.as_matrix(), kde=False, rug=True)

            print allocation_mean, allocation_mean.shape
            print conflict_deficit, conflict_deficit.shape
            data_by_res_percent[a['resource_percentage']][a['delta']] = \
                (allocation_mean, conflict_deficit)

        for resource_percentage, data in data_by_res_percent.iteritems():
            print resource_percentage
            print data
            for delta, (x, y) in data.iteritems():
                # f, ax = plt.subplots(figsize=(8, 8))
                # ax.set(yscale="log")
                sns.regplot(x, y, fit_reg=False, label="$\delta = %.5f$" % delta)
            plt.legend()
            # plt.title(u"Análise dos Conflitos para %.2f da Média de %s "
            #          % (resource_percentage, i18n[resource_type]))
            plt.xlabel(u"Média das alocações passados ao ocorrer o conflito")
            plt.ylabel(u"Déficit de %s na Alocação" % i18n[resource_type])
            plt.savefig('conflict-%s-%.5f.pdf' % (resource_type, resource_percentage),
                        bbox_inches='tight')
            plt.close()


# Tempo de execução da função allocate para vários usuários (comparado com o
# método de otimização)
def runtime_optimize_functions():
    pass


# Suavidade das alocações para os diferentes deltas e r_i
def allocation_smoothness(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        data_by_delta = defaultdict(list)
        file_name = alloc.user_type_files[resource_type]
        user_needs = pd.read_csv(file_name, index_col='time')
        num_reqs = (user_needs > 0).sum().sum()
        res_means = user_needs.mean()

        user_types_cache = {}

        def get_user_types(resource_percentage):
            if resource_percentage in user_types_cache:
                return user_types_cache[resource_percentage]
            user_resources = resource_percentage * res_means
            user_resources = user_resources.apply(
                lambda x: np.ceil(x).astype(np.int64))
            user_types = user_needs - user_resources
            user_resources[user_resources.index.str.contains('fr')] = 0
            user_types_cache[resource_percentage] = user_types
            return user_types_cache[resource_percentage]

        for a in allocations:
            allocation = pd.read_csv(a['file_name'], index_col='time')
            resource_percentage = a['resource_percentage']
            user_types = get_user_types(resource_percentage)

            # variance_relation = allocation.var()/user_types.var()
            # variance_relation = variance_relation.replace([np.inf, -np.inf], np.nan)
            # variance_relation_mean = variance_relation.mean()

            autocorrelation = allocation.apply(lambda x: x.autocorr(lag=1))
            autocorrelation_mean = autocorrelation.median()

            data_point = (resource_percentage, autocorrelation_mean)
            data_by_delta[a['delta']].append(data_point)

        f, ax = plt.subplots(figsize=(6, 3))
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        marker = itertools.cycle(('+', '.', 'o', '*', '^', 's','d'))
        for delta in sorted(data_by_delta.iterkeys()):
            points = data_by_delta[delta]
            points.sort()
            label = "$\delta = %.5f$" % delta
            ax.plot(*zip(*points), label=label, marker=marker.next())
        # plt.title(u"Análise de Suavidade para %s" % i18n[resource_type])
        plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
        plt.ylabel(u"Estabilidade")
        plt.ylim([0.0, 1])
        plt.legend(loc=4)
        sns.despine()
        plt.savefig('autocorr_median-%s.pdf' % resource_type, bbox_inches='tight')
        plt.close()


# Warning! takes too much time and produce results that are similar to the
# allocation_smoothness while hard to explain
def smoothness_by_parts(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        print resource_type
        import warnings
        warnings.simplefilter("error")

        data_by_delta = defaultdict(list)
        file_name = alloc.user_type_files[resource_type]
        user_needs = pd.read_csv(file_name, index_col='time')
        num_reqs = (user_needs > 0).sum().sum()
        res_means = user_needs.mean()

        user_types_cache = {}

        def get_user_types(resource_percentage):
            if resource_percentage in user_types_cache:
                return user_types_cache[resource_percentage]
            user_resources = resource_percentage * res_means
            user_resources = user_resources.apply(
                lambda x: np.ceil(x).astype(np.int64))
            user_types = user_needs - user_resources
            user_resources[user_resources.index.str.contains('fr')] = 0
            user_types_cache[resource_percentage] = user_types
            return user_types_cache[resource_percentage]
        def group_consecutive(data, ref):
            return np.split(data, np.where(np.diff(ref) != 0)[0] + 1)

        for a in allocations:
            allocation = pd.read_csv(a['file_name'], index_col='time')
            resource_percentage = a['resource_percentage']
            user_types = get_user_types(resource_percentage)

            auto_corr_sum_all = 0.0
            num_users = 0
            for al, tp in zip(allocation, user_types):
                auto_corr_sum = 0.0
                weights = 0
                allocation_chunks = group_consecutive(allocation[al],
                                                      user_types[tp])
                for chunk in allocation_chunks:
                    size = chunk.size
                    if chunk.size <= 2:
                        continue
                    autocorr = chunk.autocorr(lag=1)
                    if np.isnan(autocorr):
                        autocorr = 1.0
                    auto_corr_sum += autocorr*size
                    weights += size
                auto_corr_sum_all += auto_corr_sum/weights
                num_users += 1

            autocorrelation_mean = auto_corr_sum_all/num_users

            data_point = (resource_percentage, autocorrelation_mean)
            data_by_delta[a['delta']].append(data_point)

        f, ax = plt.subplots(figsize=(6, 4))
        marker = itertools.cycle(('+', '.', 'o', '*', '^', 's', 'd'))
        for delta in sorted(data_by_delta.iterkeys()):
            points = data_by_delta[delta]
            points.sort()
            label = "$\delta = %.5f$" % delta
            ax.errorbar(*zip(*points), label=label,
                        marker=marker.next())
        # plt.title(u"Análise de Suavidade para %s" % i18n[resource_type])
        plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
        plt.ylabel(u"Estabilidade Média")
        plt.ylim([0.2, 1])
        plt.legend(loc=4)
        sns.despine()
        plt.savefig('autocorr_chunk_mean-%s.pdf' % resource_type,
                    bbox_inches='tight')
        plt.close()


@click.command()
@click.argument('data_path',
                type=click.Path(exists=True, file_okay=False, readable=True))
def main(data_path):
    # request_fulfilment(data_path)
    # resource_vs_utility(data_path)
    # credibility_distribution(data_path)
    # allocation_smoothness(data_path)
    # resource_utilization_bar_plot(data_path)
    resources_received_vs_given(data_path)
    # # request_justice(data_path)
    # # conflict_resources(data_path)


if __name__ == '__main__':
    main()
