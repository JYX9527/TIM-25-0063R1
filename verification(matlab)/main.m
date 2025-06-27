clc;clear all;close all;
height=480;width=640;img_width =640; img_height = 480; 

load coeff_all.mat

 %for A = 1:20
    A=20;%% Change serial number
    fprintf('Processing sample A = %d...\n', A);

    %% 
    filenameI = sprintf('Wrapped_phase_UNet\\dataset-MD\\test\\input\\I(%d).mat', A);
    test = load(filenameI);
    test = test.a;
    %% 
    filenamem = sprintf('Wrapped_phase_UNet\\dataset-MD\\test\\output\\M(%d).mat', A);
    M = load(filenamem); m = M.M;

    filenamed = sprintf('Wrapped_phase_UNet\\dataset-MD\\test\\output\\D(%d).mat', A);
    D = load(filenamed); d = D.D;

    trueMask = sqrt(m.^2 + d.^2) / 6;
    trueMask(trueMask < 5) = 0;
    trueMask(trueMask > 0) = 1;

    % 
    deta1 = -atan2(m, d);
    deta1 = deta1 .* trueMask;

    %% 
    filenameh = sprintf('Rough_height_UNet\\dataset-H\\test\\output\\H(%d).mat', A);
    H = load(filenameh); h = double(H.H);
    h = h .* trueMask;

    % 
    h(h >= 95 | h < -2) = nan;

    %% 
    up_test_obj = h;
    [x_grid, y_grid] = meshgrid(1:img_width, 1:img_height);
    X_input = [x_grid(:), y_grid(:), up_test_obj(:)];
    phase1 = fit_fun1(param, X_input);
    p = reshape(phase1, img_height, img_width);
    pp = deta1 + 2 * pi * round((p - deta1) / (2 * pi));

    %% 
    up_test_obj = pp .* trueMask;
    X_input(:,3) = up_test_obj(:);
    height_est = fit_fun(param, X_input);
    hh = reshape(height_est, img_height, img_width);
    hh(hh >= 95 | hh < -2) = nan;

    %% 
    a0 = flipud(hh .* trueMask);

    figure(16), clf, axis off, hold on
    set(gcf, 'Color','w'); caxis([0,95]); camlight
    surf(a0, 'FaceColor','interp', 'EdgeColor','none', 'FaceLighting','phong')
    view(0,90); caxis([0,70]); camlight left
    axis([0 img_width 0 img_height 1 100])
    xlabel('X/pixel'); ylabel('Y/pixel');
    xlim([0, img_width]); ylim([0, img_height]);
    set(gca, 'LineWidth', 1, 'FontSize', 22, 'FontName', 'Times');
    set(get(gca, 'ylabel'), 'Position', [-120 200])
    set(get(gca, 'xlabel'), 'Position', [160 -80])

    % saveas(gcf, sprintf('figure_A%d.png', A));
% end

