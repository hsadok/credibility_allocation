# -*- coding: utf-8 -*-
from collections import defaultdict
import itertools
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from helpers.allocation import Allocation

i18n = {'memory': u"Memória", 'cpu': "CPU"}


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
        plt.savefig('mean_utility-%s.pdf' % resource_type, bbox_inches='tight')
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
            complete_usage_count =list(used_resources_sum).count(res_available)
            percentages.append(complete_usage_count*100.0/num_elements)
            labels.append(resource_percentage)

        ax.bar(range(1, len(percentages) + 1), percentages, tick_label=labels)

        plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
        plt.ylabel(u"Iterações (%)")
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
            contributed_resources = \
                contributed_resources[received_resources > 0]
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
        plt.savefig('autocorr_median-%s.pdf' % resource_type,
                    bbox_inches='tight')
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

        plt.xlabel(u"Quantidade de %s ($r_i$/média)" % i18n[resource_type])
        plt.ylabel(u"Estabilidade Média")
        plt.ylim([0.2, 1])
        plt.legend(loc=4)
        plt.savefig('autocorr_chunk_mean-%s.pdf' % resource_type,
                    bbox_inches='tight')
        plt.close()
