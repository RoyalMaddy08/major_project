% Time setup
t = 0:0.001:1;
dt = 0.001;
% --- EV Battery Voltage (DC source) ---
V_base = 12 - 0.01*t;
% --- Current profile (drive cycle) ---
I = 2 + 0.5*sin(2*pi*2*t);
% Acceleration
I = I + 5*(1./(1 + exp(-40*(t - 0.3))));
% Regenerative braking
I = I - 3*(1./(1 + exp(-40*(t - 0.7))));
% --- Voltage dynamics ---
R = 0.2;
alpha = 0.1;
V_dyn = zeros(size(t));
V_dyn(1) = V_base(1);
for k = 2:length(t)
    V_drop = V_base(k) - R * I(k);
    V_dyn(k) = (1 - alpha)*V_drop + alpha*V_dyn(k-1);
end
% --- Thermal model ---
tau = 0.1;
T = zeros(size(t));
T(1) = 30;

for k = 2:length(t)
    % Joule heating
    heat_input = 0.5 * I(k)^2;
    
    % Smooth thermal anomaly
    anomaly = 3 * (1 / (1 + exp(-40*(t(k) - 0.6))));
    heat_input = heat_input + anomaly;
    
    % Cooling
    cooling = (T(k-1) - 30);
    
    % Thermal dynamics
    dT = (heat_input - cooling) / tau;
    T(k) = T(k-1) + dt * dT;
    
    % Clamp temperature
    T(k) = min(T(k), 100);
end
% --- Add measurement noise ---
V_dyn = V_dyn + 0.1 * randn(size(V_dyn));
I = I + 0.05 * randn(size(I));
% --- Multiclass labeling (0=Normal, 1=Warning, 2=Critical) ---
label = zeros(size(t));
for k = 1:length(t)
    if T(k) < 35
        label(k) = 0;   % Normal
    elseif T(k) < 45
        label(k) = 1;   % Warning
    else
        label(k) = 2;   % Critical
    end
end
% --- Create dataset ---
data = [t' V_dyn' I' T' label'];
% --- Save dataset ---
headers = {'Time','Voltage','Current','Temperature','State'};
writecell(headers, 'ev_battery_dataset_multiclass.csv');
writematrix(data, 'ev_battery_dataset_multiclass.csv', 'WriteMode', 'append');
save('ev_battery_dataset_multiclass.mat', 't', 'V_dyn', 'I', 'T', 'label');
