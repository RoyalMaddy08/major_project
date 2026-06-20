// BMS Dashboard Control & Visualization Script

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const btnStart = document.getElementById('btn-start');
    const btnPause = document.getElementById('btn-pause');
    const btnStop = document.getElementById('btn-stop');
    const btnReset = document.getElementById('btn-reset');
    const cycleSelect = document.getElementById('cycle-select');
    const chemSelect = document.getElementById('chem-select');
    const agingToggle = document.getElementById('aging-toggle');
    
    const simBadge = document.getElementById('sim-status-badge');
    const dbBadge = document.getElementById('db-status-badge');
    const modelBadge = document.getElementById('model-status-badge');
    const mlWarningBanner = document.getElementById('ml-warning-banner');
    
    const valVoltage = document.getElementById('val-voltage');
    const valCurrent = document.getElementById('val-current');
    const valTemp = document.getElementById('val-temp');
    const valTime = document.getElementById('val-time');
    
    const valEkfSoc = document.getElementById('val-ekf-soc');
    const valEsnSoc = document.getElementById('val-esn-soc');
    const valEkfSoh = document.getElementById('val-ekf-soh');
    const valEsnSoh = document.getElementById('val-esn-soh');

    // Chart Handles
    let chartSOC, chartSOH, chartSOCDiff, chartSOHDiff, chartBenchTime, chartBenchMem;

    // Charts Configuration Options — Professional Light Mode
    const CHART_COLORS = {
        gridLine:   'rgba(203, 213, 225, 0.55)',   // slate-300 at low opacity
        tickLabel:  '#64748b',                      // slate-500
        axisLabel:  '#475569',                      // slate-600
        amber:      '#d97706',                      // EKF traditional
        blue:       '#2563eb',                      // ESN SOC
        violet:     '#7c3aed',                      // ESN SOH
        rose:       '#e11d48',                      // deviation lines
        slate:      '#475569',                      // EKF latency
        teal:       '#0d9488',                      // RAM footprint
    };

    const commonChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: {
                display: true,
                labels: {
                    color: CHART_COLORS.tickLabel,
                    font: { family: 'Inter', size: 11, weight: '500' },
                    boxWidth: 12,
                    boxHeight: 2,
                    padding: 14
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                title: {
                    display: true, text: 'Time (s)',
                    color: CHART_COLORS.axisLabel,
                    font: { family: 'Inter', size: 11 }
                },
                grid: { color: CHART_COLORS.gridLine, lineWidth: 1 },
                ticks: { color: CHART_COLORS.tickLabel, font: { family: 'Inter', size: 10 } },
                border: { color: '#e2e8f0' }
            },
            y: {
                grid: { color: CHART_COLORS.gridLine, lineWidth: 1 },
                ticks: { color: CHART_COLORS.tickLabel, font: { family: 'Inter', size: 10 } },
                border: { color: '#e2e8f0' }
            }
        }
    };

    // Initialize Chart.js Instances
    function initCharts() {
        // 1. SOC Chart (EKF vs. ESN)
        const ctxSOC = document.getElementById('chart-soc').getContext('2d');
        chartSOC = new Chart(ctxSOC, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'EKF (Traditional)',
                        data: [],
                        borderColor: CHART_COLORS.amber,
                        backgroundColor: 'rgba(217,119,6,0.06)',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'ESN (Reservoir ML)',
                        data: [],
                        borderColor: CHART_COLORS.blue,
                        backgroundColor: 'rgba(37,99,235,0.06)',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    }
                ]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        min: 0, max: 1.05,
                        title: { display: true, text: 'SOC Ratio', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });

        // 2. SOH Chart (EKF vs. ESN)
        const ctxSOH = document.getElementById('chart-soh').getContext('2d');
        chartSOH = new Chart(ctxSOH, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'EKF / R₀ Estimator',
                        data: [],
                        borderColor: CHART_COLORS.amber,
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'ESN (Reservoir ML)',
                        data: [],
                        borderColor: CHART_COLORS.violet,
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    }
                ]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        min: 0.0, max: 1.05,
                        title: { display: true, text: 'SOH Ratio', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });

        // 3. SOC Deviation Chart
        const ctxSOCDiff = document.getElementById('chart-soc-diff').getContext('2d');
        chartSOCDiff = new Chart(ctxSOCDiff, {
            type: 'line',
            data: {
                datasets: [{
                    label: '|EKF − ESN| SOC Error',
                    data: [],
                    borderColor: CHART_COLORS.rose,
                    borderWidth: 1.5,
                    backgroundColor: 'rgba(225,29,72,0.07)',
                    fill: true,
                    pointRadius: 0,
                    tension: 0.3
                }]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        title: { display: true, text: 'SOC Error', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });

        // 4. SOH Deviation Chart
        const ctxSOHDiff = document.getElementById('chart-soh-diff').getContext('2d');
        chartSOHDiff = new Chart(ctxSOHDiff, {
            type: 'line',
            data: {
                datasets: [{
                    label: '|EKF − ESN| SOH Error',
                    data: [],
                    borderColor: CHART_COLORS.rose,
                    borderWidth: 1.5,
                    backgroundColor: 'rgba(225,29,72,0.07)',
                    fill: true,
                    pointRadius: 0,
                    tension: 0.3
                }]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        title: { display: true, text: 'SOH Error', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });

        // 5. Execution Latency Chart
        const ctxBenchTime = document.getElementById('chart-bench-time').getContext('2d');
        chartBenchTime = new Chart(ctxBenchTime, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'EKF Step (ms)',
                        data: [],
                        borderColor: CHART_COLORS.slate,
                        borderWidth: 1.5,
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'ESN Step (ms)',
                        data: [],
                        borderColor: CHART_COLORS.blue,
                        borderWidth: 1.5,
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    }
                ]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        title: { display: true, text: 'Latency (ms)', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });

        // 6. Memory Footprint Chart
        const ctxBenchMem = document.getElementById('chart-bench-mem').getContext('2d');
        chartBenchMem = new Chart(ctxBenchMem, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Process RAM RSS (MB)',
                    data: [],
                    borderColor: CHART_COLORS.teal,
                    borderWidth: 1.5,
                    fill: true,
                    backgroundColor: 'rgba(13,148,136,0.08)',
                    pointRadius: 0,
                    tension: 0.3
                }]
            },
            options: {
                ...commonChartOptions,
                scales: {
                    ...commonChartOptions.scales,
                    y: {
                        ...commonChartOptions.scales.y,
                        title: { display: true, text: 'Memory (MB)', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 11 } }
                    }
                }
            }
        });
    }

    // Server API calls helper
    async function apiRequest(endpoint, method = 'GET', body = null) {
        try {
            const options = {
                method,
                headers: { 'Content-Type': 'application/json' }
            };
            if (body) options.body = JSON.stringify(body);
            const response = await fetch(endpoint, options);
            return await response.json();
        } catch (error) {
            console.error(`API Error on ${endpoint}:`, error);
            return null;
        }
    }

    // Refresh status and configuration parameters
    async function refreshStatus() {
        const status = await apiRequest('/api/status');
        if (!status) return;

        // Update status buttons state
        btnStart.disabled = status.sim_running;
        btnPause.disabled = !status.sim_running;
        btnStop.disabled = !status.sim_running;
        
        cycleSelect.value = status.active_cycle;
        chemSelect.value = status.chemistry;
        agingToggle.checked = status.accelerated_aging;

        // Update Simulator badge
        if (status.sim_running) {
            simBadge.querySelector('.dot').className = 'dot pulse-green';
            let labelText = "Simulator: " + status.active_cycle.toUpperCase();
            if (status.accelerated_aging) labelText += " (AGING)";
            simBadge.querySelector('.label').textContent = labelText;
        } else {
            simBadge.querySelector('.dot').className = 'dot pulse-red';
            simBadge.querySelector('.label').textContent = 'Simulator: Idle';
        }

        // Update DB badge
        if (status.mongodb_connected) {
            dbBadge.querySelector('.dot').className = 'dot pulse-green';
            dbBadge.querySelector('.label').textContent = 'MongoDB: Connected';
        } else {
            dbBadge.querySelector('.dot').className = 'dot pulse-amber';
            dbBadge.querySelector('.label').textContent = 'MongoDB: In-Memory';
        }

        // Update ESN model badge
        if (status.model_loaded) {
            modelBadge.querySelector('.dot').className = 'dot pulse-green';
            modelBadge.querySelector('.label').textContent = 'ESN Model: Active';
            mlWarningBanner.classList.add('hidden');
        } else {
            modelBadge.querySelector('.dot').className = 'dot pulse-red';
            modelBadge.querySelector('.label').textContent = 'ESN Model: Missing';
            mlWarningBanner.classList.remove('hidden');
        }
    }

    // Refresh telemetry and plot data
    async function refreshTelemetry() {
        const telemetry = await apiRequest('/api/telemetry');
        if (!telemetry || !telemetry.data) return;

        const data = telemetry.data;
        
        // Update numerical readouts with latest values
        if (data.length > 0) {
            const latest = data[data.length - 1];
            valVoltage.textContent = latest.voltage.toFixed(2);
            valCurrent.textContent = latest.current.toFixed(2);
            valTemp.textContent = latest.temperature.toFixed(1);
            valTime.textContent = Math.round(latest.time);
            
            valEkfSoc.textContent = (latest.ekf_soc * 100.0).toFixed(1) + '%';
            valEsnSoc.textContent = (latest.esn_soc * 100.0).toFixed(1) + '%';
            valEkfSoh.textContent = (latest.ekf_soh * 100.0).toFixed(1) + '%';
            valEsnSoh.textContent = (latest.esn_soh * 100.0).toFixed(1) + '%';
        } else {
            valVoltage.textContent = '0.00';
            valCurrent.textContent = '0.00';
            valTemp.textContent = '0.0';
            valTime.textContent = '0';
            
            valEkfSoc.textContent = '0.0%';
            valEsnSoc.textContent = '0.0%';
            valEkfSoh.textContent = '100.0%';
            valEsnSoh.textContent = '100.0%';
        }

        // Map data arrays
        const socEkfData = data.map(r => ({ x: r.time, y: r.ekf_soc }));
        const socEsnData = data.map(r => ({ x: r.time, y: r.esn_soc }));
        
        const sohEkfData = data.map(r => ({ x: r.time, y: r.ekf_soh }));
        const sohEsnData = data.map(r => ({ x: r.time, y: r.esn_soh }));
        
        const socDiffData = data.map(r => ({ x: r.time, y: Math.abs(r.ekf_soc - r.esn_soc) }));
        const sohDiffData = data.map(r => ({ x: r.time, y: Math.abs(r.ekf_soh - r.esn_soh) }));
        
        const latencyEkf = data.map(r => ({ x: r.time, y: r.ekf_time }));
        const latencyEsn = data.map(r => ({ x: r.time, y: r.esn_time }));
        
        const memData = data.map(r => ({ x: r.time, y: r.mem_usage }));

        // Update graph datasets
        chartSOC.data.datasets[0].data = socEkfData;
        chartSOC.data.datasets[1].data = socEsnData;
        chartSOC.update();

        chartSOH.data.datasets[0].data = sohEkfData;
        chartSOH.data.datasets[1].data = sohEsnData;
        chartSOH.update();

        chartSOCDiff.data.datasets[0].data = socDiffData;
        chartSOCDiff.update();

        chartSOHDiff.data.datasets[0].data = sohDiffData;
        chartSOHDiff.update();

        chartBenchTime.data.datasets[0].data = latencyEkf;
        chartBenchTime.data.datasets[1].data = latencyEsn;
        chartBenchTime.update();

        chartBenchMem.data.datasets[0].data = memData;
        chartBenchMem.update();
    }

    // Set up controls event listeners
    btnStart.addEventListener('click', async () => {
        await apiRequest('/api/control', 'POST', { command: 'start' });
        refreshStatus();
    });

    btnPause.addEventListener('click', async () => {
        await apiRequest('/api/control', 'POST', { command: 'stop' });
        refreshStatus();
    });

    btnStop.addEventListener('click', async () => {
        await apiRequest('/api/control', 'POST', { command: 'stop' });
        refreshStatus();
    });

    btnReset.addEventListener('click', async () => {
        if (confirm('Are you sure you want to reset the simulation? This clears current MongoDB telemetry.')) {
            await apiRequest('/api/control', 'POST', { command: 'reset' });
            refreshStatus();
            refreshTelemetry();
        }
    });

    cycleSelect.addEventListener('change', async (e) => {
        await apiRequest('/api/control', 'POST', { cycle_type: e.target.value });
        refreshStatus();
    });

    chemSelect.addEventListener('change', async (e) => {
        await apiRequest('/api/control', 'POST', { chemistry: e.target.value });
        refreshStatus();
        refreshTelemetry();
    });

    agingToggle.addEventListener('change', async (e) => {
        await apiRequest('/api/control', 'POST', { accelerated_aging: e.target.checked });
        refreshStatus();
    });

    // Run Initialization
    initCharts();
    refreshStatus();
    refreshTelemetry();

    // Start Polling loops
    setInterval(refreshStatus, 1500);
    setInterval(refreshTelemetry, 1000);
});
