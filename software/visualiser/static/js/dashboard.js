// Battery State Estimator Dashboard Control & Visualization Script

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const mismatchSelect = document.getElementById('mismatch-select');
    const quantizeSelect = document.getElementById('quantize-select');
    
    const simBadge = document.getElementById('sim-status-badge');
    const simPortBadge = document.getElementById('sim-port-badge');
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

    // Advanced Estimations selectors
    const valEkfSoe = document.getElementById('val-ekf-soe');
    const valEsnSoe = document.getElementById('val-esn-soe');
    const valEnergyRem = document.getElementById('val-energy-rem');
    const valSopDischarge = document.getElementById('val-sop-discharge');
    const valSopCharge = document.getElementById('val-sop-charge');
    const valSopCurrents = document.getElementById('val-sop-currents');
    const valEkfRul = document.getElementById('val-ekf-rul');
    const valEsnRul = document.getElementById('val-esn-rul');
    const valRulStatus = document.getElementById('val-rul-status');

    // Large top summary fields
    const valTrueSocLarge = document.getElementById('val-true-soc-large');
    const valTrueSohLarge = document.getElementById('val-true-soh-large');
    
    // Diagnostics Elements
    const diagEnvTemp = document.getElementById('diag-env-temp');
    const diagHealthRing = document.getElementById('diag-health-ring');
    const diagStatusTitle = document.getElementById('diag-status-title');
    const diagStatusDesc = document.getElementById('diag-status-desc');
    const alarmSensor = document.getElementById('alarm-sensor');
    const alarmShort = document.getElementById('alarm-short');
    const alarmThermal = document.getElementById('alarm-thermal');

    // Chart Handles
    let chartSOC, chartSOH;

    // Charts Configuration Options — Premium Light Mode
    const CHART_COLORS = {
        gridLine:   'rgba(226, 232, 240, 0.75)',    // Slate 200
        tickLabel:  '#64748b',                      // Slate 500
        axisLabel:  '#334155',                      // Slate 700
        emerald:    '#10b981',                      // True ground truth
        amber:      '#b45309',                      // EKF traditional (Dark Amber)
        blue:       '#1d4ed8',                      // ESN SOC (Royal Blue)
        violet:     '#6d28d9',                      // ESN SOH (Purple)
    };

    // Helper to build gradient fills for area charts
    function getFadedGradient(ctx, hexColor) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 180);
        const r = parseInt(hexColor.slice(1, 3), 16);
        const g = parseInt(hexColor.slice(3, 5), 16);
        const b = parseInt(hexColor.slice(5, 7), 16);
        gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.08)`);
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
                        label: 'True SOC (Reference)',
                        data: [],
                        borderColor: CHART_COLORS.emerald,
                        backgroundColor: gradientSOC,
                        borderWidth: 2.5,
                        fill: true,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'EKF + CC Estimation',
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
                        label: 'True SOH (Reference)',
                        data: [],
                        borderColor: CHART_COLORS.emerald,
                        backgroundColor: gradientSOH,
                        borderWidth: 2.5,
                        fill: true,
                        pointRadius: 0,
                        tension: 0.3
                    },
                    {
                        label: 'EKF Resistance tracker',
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

    // ── CSS-class helpers for the diagnostics panel ─────────────
    const RING_CLASSES = ['diag-ring-nominal', 'diag-ring-warning', 'diag-ring-critical'];

    function setAlarmBadge(el, isActive, activeClass) {
        if (!el) return;
        if (isActive) {
            el.classList.add(activeClass);
        } else {
            el.classList.remove(activeClass);
        }
    }

    function setHealthRing(ringEl, titleEl, descEl, ringClass, iconHtml, titleText, descText) {
        RING_CLASSES.forEach(c => ringEl.classList.remove(c));
        ringEl.classList.add(ringClass);
        ringEl.innerHTML = iconHtml;
        titleEl.textContent = titleText;
        descEl.textContent = descText;
    }

    // Refresh status and configuration parameters
    async function refreshStatus() {
        const status = await apiRequest('/api/status');
        if (!status) return;

        mismatchSelect.value = status.ekf_mismatch !== undefined ? Number(status.ekf_mismatch).toFixed(1) : "1.0";
        quantizeSelect.value = status.quantize_mode || "float32";

        // Update Ambient Temp status badge
        if (diagEnvTemp && status.T_ambient !== undefined) {
            diagEnvTemp.textContent = status.T_ambient.toFixed(1) + '°C Ambient';
        }

        // Update Simulator badge
        if (status.sim_running) {
            simBadge.querySelector('.dot').className = 'dot pulse-green';
            let labelText = "Simulator: " + status.active_cycle.toUpperCase();
            if (status.accelerated_aging) labelText += " (AGING)";
            if (status.fault_short || status.fault_thermal || status.fault_dropout) labelText += " (FAULT)";
            simBadge.querySelector('.label').textContent = labelText;
        } else {
            simBadge.querySelector('.dot').className = 'dot pulse-red';
            simBadge.querySelector('.label').textContent = 'Simulator: Idle';
        }

        // Update Port 8000/simulator service badge
        if (simPortBadge) {
            if (status.simulator_port_online) {
                simPortBadge.querySelector('.dot').className = 'dot pulse-green';
                simPortBadge.querySelector('.label').textContent = 'Sim Service: Online';
            } else {
                simPortBadge.querySelector('.dot').className = 'dot pulse-red';
                simPortBadge.querySelector('.label').textContent = 'Sim Service: Offline';
            }
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

        // Update Chemistry badge in header
        const chemBadgeLabel = document.getElementById('val-chem-badge');
        if (chemBadgeLabel && status.chemistry) {
            const chemMap = {
                'li_ion': 'Generic Li-ion',
                'nmc': 'Li-ion NMC',
                'lfp': 'LiFePO₄ LFP',
                'lead_acid': 'Lead-Acid'
            };
            const chemName = chemMap[status.chemistry] || status.chemistry.toUpperCase();
            chemBadgeLabel.textContent = 'Chemistry: ' + chemName;
        }

        // Update Simulator State Reference Card in Sidebar
        const valSimRunning = document.getElementById('val-sim-running');
        const valSimChemistry = document.getElementById('val-sim-chemistry');
        const valSimCycle = document.getElementById('val-sim-cycle');
        const valSimAmbient = document.getElementById('val-sim-ambient');
        const valSimAging = document.getElementById('val-sim-aging');

        const injShort = document.getElementById('status-injected-short');
        const injThermal = document.getElementById('status-injected-thermal');
        const injDropout = document.getElementById('status-injected-dropout');

        if (valSimRunning) {
            if (status.sim_running) {
                valSimRunning.textContent = 'Active';
                valSimRunning.style.color = '#10b981';
            } else {
                valSimRunning.textContent = 'Idle';
                valSimRunning.style.color = '#ef4444';
            }
        }

        if (valSimChemistry) {
            const chemMap = {
                'li_ion': 'Generic Li-ion',
                'nmc': 'Li-ion NMC',
                'lfp': 'LiFePO₄ LFP',
                'lead_acid': 'Lead-Acid'
            };
            valSimChemistry.textContent = chemMap[status.chemistry] || status.chemistry.toUpperCase();
        }

        if (valSimCycle) {
            const cycleMap = {
                'udds': 'UDDS Cycle',
                'hwfet': 'HWFET Cycle',
                'us06': 'US06 Cycle',
                'constant': 'Constant (-1C)',
                'charge': 'CC-CV Charge'
            };
            valSimCycle.textContent = cycleMap[status.active_cycle] || status.active_cycle.toUpperCase();
        }

        if (valSimAmbient && status.T_ambient !== undefined) {
            valSimAmbient.textContent = status.T_ambient.toFixed(1) + ' °C';
        }

        if (valSimAging) {
            if (status.accelerated_aging) {
                valSimAging.textContent = 'ON (x1500)';
                valSimAging.style.color = '#b45309';
            } else {
                valSimAging.textContent = 'OFF';
                valSimAging.style.color = 'var(--text-muted)';
            }
        }

        // Active Injected Fault indicators
        if (injShort) {
            if (status.fault_short) {
                injShort.style.color = '#be123c';
                injShort.style.fontWeight = '600';
                injShort.querySelector('i').style.color = '#be123c';
            } else {
                injShort.style.color = 'var(--text-muted)';
                injShort.style.fontWeight = 'normal';
                injShort.querySelector('i').style.color = 'var(--text-muted)';
            }
        }
        if (injThermal) {
            if (status.fault_thermal) {
                injThermal.style.color = '#be123c';
                injThermal.style.fontWeight = '600';
                injThermal.querySelector('i').style.color = '#be123c';
            } else {
                injThermal.style.color = 'var(--text-muted)';
                injThermal.style.fontWeight = 'normal';
                injThermal.querySelector('i').style.color = 'var(--text-muted)';
            }
        }
        if (injDropout) {
            if (status.fault_dropout) {
                injDropout.style.color = '#be123c';
                injDropout.style.fontWeight = '600';
                injDropout.querySelector('i').style.color = '#be123c';
            } else {
                injDropout.style.color = 'var(--text-muted)';
                injDropout.style.fontWeight = 'normal';
                injDropout.querySelector('i').style.color = 'var(--text-muted)';
            }
        }

        // Update loaded ESN model registry RMSE details in Sidebar
        const valModelSocRmse = document.getElementById('val-model-soc-rmse');
        const valModelSohRmse = document.getElementById('val-model-soh-rmse');
        if (valModelSocRmse) {
            valModelSocRmse.textContent = (status.soc_rmse !== null && status.soc_rmse !== undefined) ? status.soc_rmse.toFixed(6) : '--';
        }
        if (valModelSohRmse) {
            valModelSohRmse.textContent = (status.soh_rmse !== null && status.soh_rmse !== undefined) ? status.soh_rmse.toFixed(6) : '--';
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

            // Update Advanced Estimations values
            if (valEkfSoe) valEkfSoe.textContent = (latest.ekf_soe * 100.0).toFixed(1);
            if (valEsnSoe) valEsnSoe.textContent = latest.esn_soe !== undefined ? (latest.esn_soe * 100.0).toFixed(1) + '%' : '--%';
            if (valEnergyRem) valEnergyRem.textContent = latest.energy_remaining_wh !== undefined ? latest.energy_remaining_wh.toFixed(1) + ' Wh' : '-- Wh';
            
            if (valSopDischarge) valSopDischarge.textContent = latest.sop_discharge_pwr !== undefined ? Math.round(latest.sop_discharge_pwr) : '--';
            if (valSopCharge) valSopCharge.textContent = latest.sop_charge_pwr !== undefined ? Math.round(latest.sop_charge_pwr) + ' W' : '-- W';
            if (valSopCurrents) {
                const disCurr = latest.sop_discharge_curr !== undefined ? latest.sop_discharge_curr.toFixed(1) : '--';
                const chgCurr = latest.sop_charge_curr !== undefined ? latest.sop_charge_curr.toFixed(1) : '--';
                valSopCurrents.textContent = `${disCurr} A / ${chgCurr} A`;
            }
            
            if (valEkfRul) valEkfRul.textContent = latest.ekf_rul_cycles !== undefined ? Math.round(latest.ekf_rul_cycles) : '--';
            if (valEsnRul) valEsnRul.textContent = latest.esn_rul_cycles !== undefined ? Math.round(latest.esn_rul_cycles) + ' cyc' : '-- cyc';
            
            if (valRulStatus && latest.ekf_soh !== undefined) {
                const soh = latest.ekf_soh;
                let status = 'Excellent';
                let color = '#10b981';
                if (soh <= 0.8) {
                    status = 'Replace (EOL)';
                    color = '#be123c';
                } else if (soh <= 0.85) {
                    status = 'Fair';
                    color = '#b45309';
                } else if (soh <= 0.9) {
                    status = 'Good';
                    color = '#1d4ed8';
                }
                valRulStatus.textContent = status;
                valRulStatus.style.color = color;
            }

            // Top battery cell status values
            const trueSocVal = latest.true_soc !== undefined ? latest.true_soc : latest.cc_soc;
            const trueSohVal = latest.true_soh !== undefined ? latest.true_soh : latest.trad_soh;
            
            valTrueSocLarge.textContent = (trueSocVal * 100.0).toFixed(1) + '%';
            valTrueSohLarge.textContent = (trueSohVal * 100.0).toFixed(1) + '%';

            // Diagnostics and faults evaluation
            const faults = latest.faults || [];
            const hasSensor = faults.includes('sensor_dropout');
            const hasShort  = faults.includes('internal_short');
            const hasThermal = faults.includes('thermal_runaway');

            // Toggle alarm badge classes (CSS handles all visual states)
            setAlarmBadge(alarmSensor,  hasSensor,  'alarm-active-sensor');
            setAlarmBadge(alarmShort,   hasShort,   'alarm-active-short');
            setAlarmBadge(alarmThermal, hasThermal, 'alarm-active-thermal');

            // Update main health ring and status text
            if (diagHealthRing && diagStatusTitle && diagStatusDesc) {
                if (hasThermal || hasSensor) {
                    let desc = '';
                    if (hasSensor)   desc += 'BMS sensor connection lost (voltage flatlined). ';
                    if (hasThermal)  desc += 'Extreme cell temperature runaway detected! ';
                    if (hasShort)    desc += 'Internal micro-short circuit leakage detected. ';
                    setHealthRing(diagHealthRing, diagStatusTitle, diagStatusDesc,
                        'diag-ring-critical', '<i class="fa-solid fa-triangle-exclamation"></i>',
                        'System Status: Critical Alert', desc);
                } else if (hasShort) {
                    setHealthRing(diagHealthRing, diagStatusTitle, diagStatusDesc,
                        'diag-ring-warning', '<i class="fa-solid fa-circle-exclamation"></i>',
                        'System Status: Warning Anomaly', 'Internal micro-short circuit leakage detected.');
                } else {
                    setHealthRing(diagHealthRing, diagStatusTitle, diagStatusDesc,
                        'diag-ring-nominal', '<i class="fa-solid fa-check"></i>',
                        'System Status: Nominal', 'All estimators running normally. No active fault anomalies.');
                }
            }
        } else {
            valVoltage.textContent = '0.00';
            valCurrent.textContent = '0.00';
            valTemp.textContent = '0.0';
            valTime.textContent = '0';

            valEkfSoc.textContent = '--%';
            valEsnSoc.textContent = '--%';
            valEkfSoh.textContent = '--%';
            valEsnSoh.textContent = '--%';

            if (valEkfSoe) valEkfSoe.textContent = '--';
            if (valEsnSoe) valEsnSoe.textContent = '--%';
            if (valEnergyRem) valEnergyRem.textContent = '-- Wh';
            if (valSopDischarge) valSopDischarge.textContent = '--';
            if (valSopCharge) valSopCharge.textContent = '-- W';
            if (valSopCurrents) valSopCurrents.textContent = '-- A / -- A';
            if (valEkfRul) valEkfRul.textContent = '--';
            if (valEsnRul) valEsnRul.textContent = '-- cyc';
            if (valRulStatus) {
                valRulStatus.textContent = 'Good';
                valRulStatus.style.color = '#10b981';
            }

            valTrueSocLarge.textContent = '--%';
            valTrueSohLarge.textContent = '--%';

            setAlarmBadge(alarmSensor,  false, 'alarm-active-sensor');
            setAlarmBadge(alarmShort,   false, 'alarm-active-short');
            setAlarmBadge(alarmThermal, false, 'alarm-active-thermal');

            if (diagHealthRing && diagStatusTitle && diagStatusDesc) {
                setHealthRing(diagHealthRing, diagStatusTitle, diagStatusDesc,
                    'diag-ring-nominal', '<i class="fa-solid fa-check"></i>',
                    'System Status: Nominal', 'All estimators running normally. No active fault anomalies.');
            }
        }

        // Map data arrays for graphs
        const socTrueData = data.map(r => ({ x: r.time, y: r.true_soc !== undefined ? r.true_soc : r.cc_soc }));
        const socEkfData = data.map(r => ({ x: r.time, y: r.ekf_soc }));
        const socEsnData = data.map(r => ({ x: r.time, y: r.esn_soc }));
        
        const sohTrueData = data.map(r => ({ x: r.time, y: r.true_soh !== undefined ? r.true_soh : r.trad_soh }));
        const sohEkfData = data.map(r => ({ x: r.time, y: r.ekf_soh }));
        const sohEsnData = data.map(r => ({ x: r.time, y: r.esn_soh }));

        // Update graph datasets
        chartSOC.data.datasets[0].data = socTrueData;
        chartSOC.data.datasets[1].data = socEkfData;
        chartSOC.data.datasets[2].data = socEsnData;
        chartSOC.update();

        chartSOH.data.datasets[0].data = sohTrueData;
        chartSOH.data.datasets[1].data = sohEkfData;
        chartSOH.data.datasets[2].data = sohEsnData;
        chartSOH.update();
    }

    mismatchSelect.addEventListener('change', async (e) => {
        await apiRequest('/api/control', 'POST', { ekf_mismatch: parseFloat(e.target.value) });
        refreshStatus();
        refreshTelemetry();
    });

    quantizeSelect.addEventListener('change', async (e) => {
        await apiRequest('/api/control', 'POST', { quantize_mode: e.target.value });
        refreshStatus();
        refreshTelemetry();
    });

    // ──────────────── ESN Model Retraining Bindings ────────────────
    const btnRetrain = document.getElementById('btn-retrain');
    const consoleEl = document.getElementById('train-console');
    let isTraining = false;

    async function pollTrainingStatus() {
        const res = await apiRequest('/api/train/status');
        if (!res) return;

        if (res.status === 'running') {
            isTraining = true;
            if (btnRetrain) {
                btnRetrain.disabled = true;
                btnRetrain.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training…';
            }
            if (consoleEl) {
                consoleEl.textContent = res.logs;
                consoleEl.scrollTop = consoleEl.scrollHeight;
            }
            setTimeout(pollTrainingStatus, 1500);
        } else {
            if (isTraining) {
                isTraining = false;
                if (btnRetrain) {
                    btnRetrain.disabled = false;
                    btnRetrain.innerHTML = '<i class="fa-solid fa-arrows-spin"></i> Retrain ESN Weights';
                }
                refreshStatus();
                refreshTelemetry();
            }
            if (consoleEl && res.status !== 'idle') {
                consoleEl.textContent = res.logs;
                consoleEl.scrollTop = consoleEl.scrollHeight;
            }
            if (res.status === 'completed') {
                document.getElementById('val-model-soc-rmse').textContent = res.soc_rmse.toFixed(6);
                document.getElementById('val-model-soh-rmse').textContent = res.soh_rmse.toFixed(6);
            } else if (res.status === 'failed') {
                document.getElementById('val-model-soc-rmse').textContent = 'ERROR';
                document.getElementById('val-model-soh-rmse').textContent = 'ERROR';
            }
        }
    }

    if (btnRetrain) {
        btnRetrain.addEventListener('click', async () => {
            if (confirm('Launch Echo State Network model retraining? This process runs in the background.')) {
                btnRetrain.disabled = true;
                btnRetrain.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Starting…';
                const res = await apiRequest('/api/train', 'POST');
                if (res && (res.status === 'started' || res.status === 'running')) {
                    pollTrainingStatus();
                } else {
                    btnRetrain.disabled = false;
                    btnRetrain.innerHTML = '<i class="fa-solid fa-arrows-spin"></i> Retrain ESN Weights';
                }
            }
        });
    }

    // Run Initialization
    initCharts();
    refreshStatus();
    refreshTelemetry();
    pollTrainingStatus();

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
