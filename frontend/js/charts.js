/**
 * Charts Module - Traffic Visualization with Chart.js
 */

const ChartsModule = (() => {
    let congestionChart = null;
    let trendChart = null;

    // Chart.js global defaults for dark theme
    function setDefaults() {
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.1)';
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.font.size = 11;
    }

    /**
     * Create or update the congestion bar chart
     */
    function renderCongestionChart(predictions) {
        setDefaults();

        const ctx = document.getElementById('congestionChart');
        if (!ctx) return;

        const labels = [];
        const data = [];
        const colors = [];

        // Sort by congestion level
        const sorted = Object.entries(predictions).sort((a, b) => b[1].congestion - a[1].congestion);

        sorted.forEach(([nodeId, info]) => {
            labels.push(nodeId);
            data.push((info.congestion * 100).toFixed(1));

            if (info.congestion > 0.6) {
                colors.push('rgba(239, 68, 68, 0.8)');
            } else if (info.congestion > 0.3) {
                colors.push('rgba(245, 158, 11, 0.8)');
            } else {
                colors.push('rgba(16, 185, 129, 0.8)');
            }
        });

        if (congestionChart) {
            congestionChart.data.labels = labels;
            congestionChart.data.datasets[0].data = data;
            congestionChart.data.datasets[0].backgroundColor = colors;
            congestionChart.update();
            return;
        }

        congestionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Congestion %',
                    data: data,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1,
                    borderRadius: 4,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(17, 24, 39, 0.95)',
                        borderColor: 'rgba(148, 163, 184, 0.2)',
                        borderWidth: 1,
                        titleFont: { weight: '600' },
                        padding: 10,
                        cornerRadius: 8,
                        callbacks: {
                            label: (ctx) => `Congestion: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(148,163,184,0.06)' },
                        ticks: {
                            callback: (v) => v + '%',
                            font: { size: 10 }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: {
                            font: { size: 10, weight: '600', family: "'JetBrains Mono', monospace" }
                        }
                    }
                },
                animation: {
                    duration: 800,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    /**
     * Create or update the traffic trend line chart
     */
    function renderTrendChart() {
        setDefaults();

        const ctx = document.getElementById('trendChart');
        if (!ctx) return;

        const hours = ['6AM', '8AM', '10AM', '12PM', '2PM', '4PM', '6PM', '8PM', '10PM'];

        // Simulated trend data for key nodes
        const datasets = [
            {
                label: 'Central Hub',
                data: [25, 78, 55, 45, 50, 72, 85, 40, 20],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 6,
                borderWidth: 2
            },
            {
                label: 'South Gate',
                data: [30, 72, 48, 48, 52, 68, 82, 35, 22],
                borderColor: '#ef4444',
                backgroundColor: 'rgba(239, 68, 68, 0.05)',
                tension: 0.4,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 6,
                borderWidth: 2
            },
            {
                label: 'West Highway',
                data: [10, 55, 35, 32, 38, 55, 62, 28, 8],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.05)',
                tension: 0.4,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 6,
                borderWidth: 2
            }
        ];

        if (trendChart) {
            trendChart.data.datasets = datasets;
            trendChart.update();
            return;
        }

        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: hours,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            boxWidth: 12,
                            boxHeight: 3,
                            usePointStyle: false,
                            padding: 12,
                            font: { size: 10 }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(17, 24, 39, 0.95)',
                        borderColor: 'rgba(148, 163, 184, 0.2)',
                        borderWidth: 1,
                        titleFont: { weight: '600' },
                        padding: 10,
                        cornerRadius: 8,
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(148,163,184,0.06)' },
                        ticks: {
                            callback: (v) => v + '%',
                            font: { size: 10 }
                        }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 10 } }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }

    /**
     * Destroy charts for cleanup
     */
    function destroy() {
        if (congestionChart) { congestionChart.destroy(); congestionChart = null; }
        if (trendChart) { trendChart.destroy(); trendChart = null; }
    }

    return {
        renderCongestionChart,
        renderTrendChart,
        destroy
    };
})();
