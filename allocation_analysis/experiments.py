# -*- coding: utf-8 -*-

# Since we have the types and allocations we can make lots of cool experiments
from collections import defaultdict
import click
import math
from scipy.stats.stats import pearsonr

import pandas as pd
import numpy as np
from os import listdir, path
# import matplotlib

# matplotlib.use('Agg')


#### Dummy to replace seaborn
from helpers.allocation import Allocation


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



sns.set_palette(sns.color_palette("Paired", 10))


i18n = {'memory': u"Memória", 'cpu': "CPU"}

sns.set_style('ticks')


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
                                 ).sum().sum() * 1.0 / num_reqs
            data_point = (resource_percentage, attended_requests)
            attended_requests_by_delta[-1.0].append(data_point)
            return user_types_cache[resource_percentage]

        for a in allocations:
            allocation = pd.read_csv(a['file_name'], index_col='time')
            resource_percentage = a['resource_percentage']
            user_types = get_user_types(resource_percentage)

            attended_requests = (allocation[user_needs > 0] >=
                                 user_types[user_needs > 0]
                                 ).sum().sum() * 1.0 / num_reqs

            data_point = (resource_percentage, attended_requests)
            attended_requests_by_delta[a['delta']].append(data_point)
        f, ax = plt.subplots(figsize=(6, 4))
        for delta in sorted(attended_requests_by_delta.iterkeys()):
            print delta
            points = attended_requests_by_delta[delta]
            points.sort()
            print zip(*points)
            if delta == -1:
                label = u'Sem o mecanismo'
            else:
                label = "$\delta = %.5f$" % delta
            plt.plot(*zip(*points), label=label)
        # plt.title(u"Necessidades de %s Satisfeitas" % i18n[resource_type])
        plt.xlabel(u"Parcela da Média de Recursos Atribuídos")
        plt.ylabel(u"Parcela das Necessidades Satisfeitas")
        plt.legend(loc=4)
        sns.despine()
        plt.savefig('fulfilment-%s.pdf' % resource_type, bbox_inches='tight')
        plt.close()
        f, ax = plt.subplots(figsize=(6, 4))
        ax.set_ylim([0.3, 1])
        for delta in sorted(attended_requests_by_delta.iterkeys()):
            if delta == -1:
                continue
            print delta
            points = attended_requests_by_delta[delta]
            points.sort()
            print zip(*points)
            label = "$\delta = %.5f$" % delta
            plt.plot(*zip(*points), label=label)
        # plt.title(u"Necessidades de %s Satisfeitas" % i18n[resource_type])
        plt.xlabel(u"Parcela da Média de Recursos Atribuídos")
        plt.ylabel(u"Parcela das Necessidades Satisfeitas")
        plt.legend(loc=4)
        sns.despine()
        plt.savefig('fulfilment_compare_delta-%s.pdf' % resource_type, bbox_inches='tight')
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
        plt.xlabel(u"Parcela da Média de Recursos Atribuídos")
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
def resource_utilization_histogram(data_path):
    alloc = Allocation(data_path)

    for resource_type, allocations in alloc.iteritems():
        file_name = alloc.user_type_files[resource_type]
        user_needs = pd.read_csv(file_name, index_col='time')
        res_means = user_needs.mean()

        df_list = []

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
            used_resources_sum = used_resources.sum(axis=1)
            print used_resources_sum.as_matrix()
            df = pd.DataFrame({'used_resources': used_resources_sum.as_matrix()
                               })
            # df['delta'] = a['delta']
            df['resource_percentage'] = a['resource_percentage']
            df_list.append(df)

        df = pd.concat(df_list)

        f, ax = plt.subplots(figsize=(6, 4))

        ax = sns.stripplot(x='resource_percentage', y='used_resources',
                           data=df, palette='GnBu_d', size=1, jitter=True, rasterized=True)
        # plt.title(u"Distribuição de %s Utilizada" %
        #          i18n[resource_type])
        plt.xlabel(u"Parcela da Média de Recursos Atribuídos")
        plt.ylabel(u"Total de Recursos Utilizados na Iteração")
        plt.legend(title=u'Recursos', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
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
            df_list.append(pd.DataFrame(
                [[corr, delta, resource_percentage]], #- 0.01*math.log10(1-delta) - 0.025]],
                columns=['corr', 'delta', 'resource_percentage'])
            )
        df = pd.concat(df_list)
        g = sns.lmplot(x='resource_percentage', size=3, aspect=2.5, y='corr', hue='delta', data=df,
                       fit_reg=False, legend=False)
        g.set_axis_labels(u'Parcela da Média de Recursos Atribuídos',
                            u'Correlação de Pearson')
        # plt.title(u'Correlação entre %s Contribuída e Recebida' %
        #          i18n[resource_type])
        plt.xlim([0.0,1.6])
        plt.legend(title=u'$\delta$', bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0.)
        sns.despine()
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
            plt.xlabel(u"Parcela da Média de Recursos Atribuídos")
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
        f, ax = plt.subplots(figsize=(6, 4))
        for delta in sorted(data_by_delta.iterkeys()):
            print delta
            points = data_by_delta[delta]
            points.sort()
            print zip(*points)
            label = "$\delta = %.5f$" % delta
            plt.plot(*zip(*points), label=label)
        # plt.title(u"Análise de Suavidade para %s" % i18n[resource_type])
        plt.xlabel(u"Parcela da Média de Recursos Atribuídos")
        plt.ylabel(u"Mediana da Autocorrelação lag-1")
        plt.legend(loc=4)
        sns.despine()
        plt.savefig('autocorr_median-%s.pdf' % resource_type, bbox_inches='tight')
        plt.close()


@click.command()
@click.argument('data_path',
                type=click.Path(exists=True, file_okay=False, readable=True))
def main(data_path):
    request_fulfilment(data_path)
    # resource_vs_utility(data_path)
    # credibility_distribution(data_path)
    # allocation_smoothness(data_path)
    # resource_utilization_histogram(data_path)
    # resources_received_vs_given(data_path)
    # # request_justice(data_path)
    # # conflict_resources(data_path)


if __name__ == '__main__':
    main()
