import numpy as np


class PerformanceData():
    def __init__(self, metrics, df, bins, bin_type, percentiles):
        self.metrics = metrics
        self.df = df
        self.bins = bins
        self.bin_type = bin_type
        self.percentiles = percentiles
        self.bin_centers = [ibin.mid for ibin in self.bins]
        self.bin_widths = [ibin.length / 2 for ibin in self.bins]
        self.performances_dict = {}
        performances = self.create_performance_data()

    def convert_iqr_to_sigma(self, quartiles, e_quartiles):
        factor = 1 / 1.349
        sigma = np.abs(quartiles[1] - quartiles[0]) * factor
        e_sigma = factor * np.sqrt(e_quartiles[0]**2 + e_quartiles[1]**2)        
        return sigma, e_sigma

    def estimate_percentile(self, data, percentiles, n_bootstraps=1000):
        data = np.array(data)
        n = data.shape[0]
        data.sort()
        i_means, means = [], []
        i_plussigmas, plussigmas = [], []
        i_minussigmas, minussigmas = [], []
        for percentile in percentiles:
            sigma = np.sqrt(percentile * n * (1 - percentile))
            mean = n * percentile
            i_means.append(int(mean))
            i_plussigmas.append(int(mean + sigma + 1))
            i_minussigmas.append(int(mean - sigma))
        bootstrap_indices = np.random.choice(
            np.arange(0, n),
            size=(n, n_bootstraps)
        )
        bootstrap_indices.sort(axis=0)
        bootstrap_samples = data[bootstrap_indices]
        for i in range(len(i_means)):
            try:
                mean = bootstrap_samples[i_means[i], :]
                means.append(np.mean(mean))
                plussigma = bootstrap_samples[i_plussigmas[i], :]
                plussigmas.append(np.mean(plussigma))
                minussigma = bootstrap_samples[i_minussigmas[i], :]
                minussigmas.append(np.mean(minussigma))
            except IndexError:
                means.append(np.nan)
                plussigmas.append(np.nan)
                minussigmas.append(np.nan)
        return means, plussigmas, minussigmas

    def calculate_performance(self, values, percentiles):
        means, plussigmas, minussigmas = self.estimate_percentile(
            values,
            percentiles
        )
        e_quartiles = []
        e_quartiles.append((plussigmas[0] - minussigmas[0]) / 2)
        e_quartiles.append((plussigmas[1] - minussigmas[1]) / 2)
        sigma, e_sigma = self.convert_iqr_to_sigma(means, e_quartiles)
        if e_sigma != e_sigma:
            sigma = np.nan
        return sigma, e_sigma

    def create_performance_data(self):
        for metric in self.metrics:
            self.performances_dict[metric] = {}
            own_performances = []
            opponent_performances = []
            own_sigmas = []
            opponent_sigmas = []
            own_lows = []
            own_medians = []
            own_highs = []
            own_centers = []
            opponent_lows = []
            opponent_medians = []
            opponent_highs = []
            opponent_centers = []
            for ibin in self.bins:
                indexer = (self.df[self.bin_type + '_binned'] == ibin)
                own_performance_temp = self.calculate_performance(
                    self.df[indexer]['own_' + metric + '_error'].values,
                    self.percentiles
                )
                own_performances.append(own_performance_temp[0])
                own_sigmas.append(own_performance_temp[1])
                own_lows.append(
                    self.df[indexer]['own_' + metric + '_error'].quantile(0.16)
                )
                own_lows.append(
                    self.df[indexer]['own_' + metric + '_error'].quantile(0.16)
                )
                own_medians.append(
                    self.df[indexer]['own_' + metric + '_error'].quantile(0.5)
                )
                own_medians.append(
                    self.df[indexer]['own_' + metric + '_error'].quantile(0.5)
                )
                own_highs.append(
                    self.df[indexer]['own_' + metric + '_error'].quantile(0.84)
                )
                own_highs.append(
                    self.df[indexer]['own_' + metric + '_error'].quantile(0.84)
                )
                own_centers.append(ibin.left)
                own_centers.append(ibin.right)
                opponent_performance_temp = self.calculate_performance(
                    self.df[indexer]['opponent_' + metric + '_error'].values,
                    self.percentiles
                )
                opponent_performances.append(opponent_performance_temp[0])
                opponent_sigmas.append(opponent_performance_temp[1])
                opponent_lows.append(
                    self.df[indexer]['opponent_' + metric + '_error'].quantile(0.16)
                )
                opponent_lows.append(
                    self.df[indexer]['opponent_' + metric + '_error'].quantile(0.16)
                )
                opponent_medians.append(
                    self.df[indexer]['opponent_' + metric + '_error'].quantile(0.5)
                )
                opponent_medians.append(
                    self.df[indexer]['opponent_' + metric + '_error'].quantile(0.5)
                )
                opponent_highs.append(
                    self.df[indexer]['opponent_' + metric + '_error'].quantile(0.84)
                )
                opponent_highs.append(
                    self.df[indexer]['opponent_' + metric + '_error'].quantile(0.84)
                )
                opponent_centers.append(ibin.left)
                opponent_centers.append(ibin.right)
            relative_improvement = np.divide(
                np.array(own_performances) - np.array(opponent_performances),
                np.array(opponent_performances)
            )
            term1 = (np.array(own_sigmas) / np.array(opponent_performances))**2
            term2 = (np.array(opponent_sigmas) * np.array(own_performances) / np.array(opponent_performances)**2)**2
            relative_improvement_sigmas = np.sqrt(term1 + term2)
            self.performances_dict[metric]['own_performances'] = own_performances
            self.performances_dict[metric]['opponent_performances'] = opponent_performances
            self.performances_dict[metric]['own_sigmas'] = own_sigmas
            self.performances_dict[metric]['opponent_sigmas'] = opponent_sigmas
            self.performances_dict[metric]['relative_improvement'] = relative_improvement
            self.performances_dict[metric]['relative_improvement_sigmas'] = relative_improvement_sigmas
            self.performances_dict[metric]['own_lows'] = own_lows
            self.performances_dict[metric]['own_medians'] = own_medians
            self.performances_dict[metric]['own_highs'] = own_highs
            self.performances_dict[metric]['own_centers'] = own_centers
            self.performances_dict[metric]['opponent_lows'] = opponent_lows
            self.performances_dict[metric]['opponent_medians'] = opponent_medians
            self.performances_dict[metric]['opponent_highs'] = opponent_highs
            self.performances_dict[metric]['opponent_centers'] = opponent_centers
