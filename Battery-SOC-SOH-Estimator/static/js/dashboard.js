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
    const mismatchSelect = document.getElementById('mismatch-select');
    const quantizeSelect = document.getElementById('quantize-select');
    
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

    // Large top summary fields
    const valTrueSocLarge = document.getElementById('val-true-soc-large');
    const valTrueSohLarge = document.getElementById('val-true-soh-large');
    
    // Battery Visualizer elements
    const batteryFluid = document.getElementById('battery-fluid-level');
    const batteryBolt = document.getElementById('battery-charging-bolt');
    const thermalGlow = document.getElementById('battery-thermal-glow');
    const bubbleContainer = document.getElementById('battery-bubbles-container');

    // Chart Handles
    let chartSOC, chartSOH, chartSOCDiff, chartSOHDiff, chartBenchTime, chartBenchMem;

    // Charts Configuration Options — Premium Light Mode
    const CHART_COLORS = {
        gridLine:   'rgba(226, 232, 240, 0.75)',    // Slate 200
        tickLabel:  '#64748b',                      // Slate 500
        axisLabel:  '#334155',                      // Slate 700
        emerald:    '#10b981',                      // True ground truth
        amber:      '#b45309',                      // EKF traditional (Dark Amber)
        blue:       '#1d4ed8',                      // ESN SOC (Royal Blue)
        violet:     '#6d28d9',                      // ESN SOH (Purple)
        rose:       '#be123c',                      // Error lines
        slate:      '#475569',                      // Bench traditional
        teal:       '#0f766e',                      // Bench memory
    };

    // Helper to build gradient fills for area charts
    function getFadedGradient(ctx, hexColor) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 180);
        const r = parseInt(hexColor.slice(1, 3), 16);
        const g = parseInt(hexColor.slice(3, 5), 16);
        const b = parseInt(hexColor.slice(5, 7), 16);
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.12)`);
        gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0.0)`);
        return gradient;
    }

    const commonChartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: {
                display: true,
                labels: {
                    color: CHART_COLORS.tickLabel,
                    font: { family: 'Inter', size: 10, weight: '600' },
                    boxWidth: 10,
                    boxHeight: 3,
                    padding: 10
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                title: {
                    display: true, text: 'Time (s)',
                    color: CHART_COLORS.axisLabel,
                    font: { family: 'Inter', size: 10, weight: '600' }
                },
                grid: { color: CHART_COLORS.gridLine, lineWidth: 1 },
                ticks: { color: CHART_COLORS.tickLabel, font: { family: 'Inter', size: 9 } },
                border: { color: '#e2e8f0' }
            },
            y: {
                grid: { color: CHART_COLORS.gridLine, lineWidth: 1 },
                ticks: { color: CHART_COLORS.tickLabel, font: { family: 'Inter', size: 9 } },
                border: { color: '#e2e8f0' }
            }
        }
    };

    // Initialize Chart.js Instances
    function initCharts() {
        // 1. SOC Chart (True vs. EKF vs. ESN)
        const ctxSOC = document.getElementById('chart-soc').getContext('2d');
        const gradientSOC = getFadedGradient(ctxSOC, CHART_COLORS.emerald);
        
        chartSOC = new Chart(ctxSOC, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'True SOC',
                        data: [],
                        borderColor: CHART_COLORS.emerald,
                        backgroundColor: gradientSOC,
                        borderWidth: 2.5,
                        fill: true,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'EKF (Traditional)',
                        data: [],
                        borderColor: CHART_COLORS.amber,
                        borderWidth: 1.8,
                        borderDash: [4, 4],
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'ESN (Reservoir ML)',
                        data: [],
                        borderColor: CHART_COLORS.blue,
                        borderWidth: 1.8,
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
                        title: { display: true, text: 'SOC Ratio', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 10 } }
                    }
                }
            }
        });

        // 2. SOH Chart (True vs. EKF vs. ESN)
        const ctxSOH = document.getElementById('chart-soh').getContext('2d');
        const gradientSOH = getFadedGradient(ctxSOH, CHART_COLORS.emerald);

        chartSOH = new Chart(ctxSOH, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'True SOH',
                        data: [],
                        borderColor: CHART_COLORS.emerald,
                        backgroundColor: gradientSOH,
                        borderWidth: 2.5,
                        fill: true,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'EKF / R₀ Estimator',
                        data: [],
                        borderColor: CHART_COLORS.amber,
                        borderWidth: 1.8,
                        borderDash: [4, 4],
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'ESN (Reservoir ML)',
                        data: [],
                        borderColor: CHART_COLORS.violet,
                        borderWidth: 1.8,
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
                        title: { display: true, text: 'SOH Ratio', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 10 } }
                    }
                }
            }
        });

        // 3. SOC Deviation Chart (EKF and ESN Error vs. True)
        const ctxSOCDiff = document.getElementById('chart-soc-diff').getContext('2d');
        chartSOCDiff = new Chart(ctxSOCDiff, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'EKF Error |True - EKF|',
                        data: [],
                        borderColor: CHART_COLORS.rose,
                        borderWidth: 1.5,
                        borderDash: [3, 3],
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'ESN Error |True - ESN|',
                        data: [],
                        borderColor: CHART_COLORS.blue,
                        borderWidth: 1.8,
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
                        title: { display: true, text: 'SOC Error', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 10 } }
                    }
                }
            }
        });

        // 4. SOH Deviation Chart (EKF and ESN Error vs. True)
        const ctxSOHDiff = document.getElementById('chart-soh-diff').getContext('2d');
        chartSOHDiff = new Chart(ctxSOHDiff, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'EKF Error |True - EKF|',
                        data: [],
                        borderColor: CHART_COLORS.rose,
                        borderWidth: 1.5,
                        borderDash: [3, 3],
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'ESN Error |True - ESN|',
                        data: [],
                        borderColor: CHART_COLORS.violet,
                        borderWidth: 1.8,
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
                        title: { display: true, text: 'SOH Error', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 10 } }
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
                        title: { display: true, text: 'Latency (ms)', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 10 } }
                    }
                }
            }
        });

        // 6. Memory Footprint Chart
        const ctxBenchMem = document.getElementById('chart-bench-mem').getContext('2d');
        const gradientMem = getFadedGradient(ctxBenchMem, CHART_COLORS.teal);
        
        chartBenchMem = new Chart(ctxBenchMem, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Process RAM RSS (MB)',
                    data: [],
                    borderColor: CHART_COLORS.teal,
                    borderWidth: 1.5,
                    fill: true,
                    backgroundColor: gradientMem,
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
                        title: { display: true, text: 'Memory (MB)', color: CHART_COLORS.axisLabel, font: { family: 'Inter', size: 10 } }
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
        mismatchSelect.value = status.ekf_mismatch !== undefined ? Number(status.ekf_mismatch).toFixed(1) : "1.0";
        quantizeSelect.value = status.quantize_mode || "float32";

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

    // Generate bubble particles inside the battery visualizer
    function generateBubbles(container, count) {
        if (!container) return;
        if (container.children.length >= count) return;
        const numToAdd = count - container.children.length;
        for (let i = 0; i < numToAdd; i++) {
            const bubble = document.createElement('div');
            bubble.className = 'battery-bubble';
            bubble.style.left = `${Math.random() * 92}%`;
            bubble.style.animationDuration = `${1.2 + Math.random() * 1.8}s`;
            bubble.style.animationDelay = `${Math.random() * 1.5}s`;
            container.appendChild(bubble);
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

            // Top battery cell status values
            const trueSocVal = latest.true_soc !== undefined ? latest.true_soc : latest.cc_soc;
            const trueSohVal = latest.true_soh !== undefined ? latest.true_soh : latest.trad_soh;
            
            valTrueSocLarge.textContent = (trueSocVal * 100.0).toFixed(1) + '%';
            valTrueSohLarge.textContent = (trueSohVal * 100.0).toFixed(1) + '%';

            // Update physical battery wave width
            batteryFluid.style.width = `${Math.max(2, trueSocVal * 100.0)}%`;

            // Adjust battery fluid color and background glows dynamically
            if (trueSocVal > 0.6) {
                batteryFluid.style.background = 'linear-gradient(90deg, #10b981 0%, #059669 100%)';
                thermalGlow.style.backgroundColor = '#10b981';
            } else if (trueSocVal > 0.2) {
                batteryFluid.style.background = 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)';
                thermalGlow.style.backgroundColor = '#f59e0b';
            } else {
                batteryFluid.style.background = 'linear-gradient(90deg, #ef4444 0%, #dc2626 100%)';
                thermalGlow.style.backgroundColor = '#ef4444';
            }

            // Manage charging bolt and bubble particles based on current direction
            const activeCurrent = latest.current; // simulator uses negative for discharge, positive for charge (adjusted in record)
            if (activeCurrent < -0.05) { // In telemetry, record sets 'current' to -I_meas (so negative is charge, positive is discharge)
                // Telemetry record sets 'current' to -I_meas.
                // In simulator, negative current means discharge. Since telemetry flips it, positive in telemetry means discharge.
                // Thus, negative current in telemetry means charging!
                batteryBolt.classList.add('active');
                generateBubbles(bubbleContainer, Math.min(8, Math.floor(Math.abs(activeCurrent) * 2)));
            } else {
                batteryBolt.classList.remove('active');
                bubbleContainer.innerHTML = '';
            }

            // Manage thermal heat alerts
            const activeTemp = latest.temperature;
            if (activeTemp > 35.0) {
                const heatScale = Math.min(1.0, (activeTemp - 35.0) / 20.0);
                thermalGlow.style.opacity = `${0.12 + heatScale * 0.38}`;
                if (activeTemp > 45.0) {
                    thermalGlow.style.backgroundColor = '#ef4444'; // Red thermal alert
                }
            } else {
                thermalGlow.style.opacity = '0.12';
            }
        } else {
            valVoltage.textContent = '0.00';
            valCurrent.textContent = '0.00';
            valTemp.textContent = '0.0';
            valTime.textContent = '0';
            
            valEkfSoc.textContent = '0.0%';
            valEsnSoc.textContent = '0.0%';
            valEkfSoh.textContent = '100.0%';
            valEsnSoh.textContent = '100.0%';

            valTrueSocLarge.textContent = '100.0%';
            valTrueSohLarge.textContent = '100.0%';
            batteryFluid.style.width = '100%';
            batteryFluid.style.background = 'linear-gradient(90deg, #10b981 0%, #059669 100%)';
            thermalGlow.style.backgroundColor = '#10b981';
            thermalGlow.style.opacity = '0.12';
            batteryBolt.classList.remove('active');
            bubbleContainer.innerHTML = '';
        }

        // Map data arrays for graphs
        const socTrueData = data.map(r => ({ x: r.time, y: r.true_soc !== undefined ? r.true_soc : r.cc_soc }));
        const socEkfData = data.map(r => ({ x: r.time, y: r.ekf_soc }));
        const socEsnData = data.map(r => ({ x: r.time, y: r.esn_soc }));
        
        const sohTrueData = data.map(r => ({ x: r.time, y: r.true_soh !== undefined ? r.true_soh : r.trad_soh }));
        const sohEkfData = data.map(r => ({ x: r.time, y: r.ekf_soh }));
        const sohEsnData = data.map(r => ({ x: r.time, y: r.esn_soh }));
        
        // Deviation errors relative to True
        const socDiffEkf = data.map(r => ({ x: r.time, y: Math.abs((r.true_soc !== undefined ? r.true_soc : r.cc_soc) - r.ekf_soc) }));
        const socDiffEsn = data.map(r => ({ x: r.time, y: Math.abs((r.true_soc !== undefined ? r.true_soc : r.cc_soc) - r.esn_soc) }));
        
        const sohDiffEkf = data.map(r => ({ x: r.time, y: Math.abs((r.true_soh !== undefined ? r.true_soh : r.trad_soh) - r.ekf_soh) }));
        const sohDiffEsn = data.map(r => ({ x: r.time, y: Math.abs((r.true_soh !== undefined ? r.true_soh : r.trad_soh) - r.esn_soh) }));
        
        const latencyEkf = data.map(r => ({ x: r.time, y: r.ekf_time }));
        const latencyEsn = data.map(r => ({ x: r.time, y: r.esn_time }));
        
        const memData = data.map(r => ({ x: r.time, y: r.mem_usage }));

        // Update graph datasets
        chartSOC.data.datasets[0].data = socTrueData;
        chartSOC.data.datasets[1].data = socEkfData;
        chartSOC.data.datasets[2].data = socEsnData;
        chartSOC.update();

        chartSOH.data.datasets[0].data = sohTrueData;
        chartSOH.data.datasets[1].data = sohEkfData;
        chartSOH.data.datasets[2].data = sohEsnData;
        chartSOH.update();

        chartSOCDiff.data.datasets[0].data = socDiffEkf;
        chartSOCDiff.data.datasets[1].data = socDiffEsn;
        chartSOCDiff.update();

        chartSOHDiff.data.datasets[0].data = sohDiffEkf;
        chartSOHDiff.data.datasets[1].data = sohDiffEsn;
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

    mismatchSelect.addEventListener('change', async (e) => {
        await apiRequest('/api/control', 'POST', { ekf_mismatch: parseFloat(e.target.value) });
        refreshStatus();
    });

    quantizeSelect.addEventListener('change', async (e) => {
        await apiRequest('/api/control', 'POST', { quantize_mode: e.target.value });
        refreshStatus();
    });

    // Run Initialization
    initCharts();
    refreshStatus();
    refreshTelemetry();

    // Start Polling loops using recursive setTimeout to prevent request stacking
    async function statusPoll() {
        await refreshStatus();
        setTimeout(statusPoll, 1500);
    }

    async function telemetryPoll() {
        await refreshTelemetry();
        setTimeout(telemetryPoll, 1000);
    }

    statusPoll();
    telemetryPoll();
});
