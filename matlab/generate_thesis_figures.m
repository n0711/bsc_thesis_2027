function generate_thesis_figures(project_root)
% MATLAB R2020a+ required for exportgraphics
% Plot recorded synthetic surrogate results, not real LLM outcomes.
% Invoke from the repository root with:
%   generate_thesis_figures(pwd)
if nargin < 1
    project_root = fileparts(fileparts(mfilename('fullpath')))
end
source_csv = fullfile(project_root,'matlab','synthetic_summary.csv')
out_dir = fullfile(project_root,'tex','figures')
if ~exist(out_dir,'dir')
    mkdir(out_dir)
end
T = readtable(source_csv)
S = T(T.length == 3,:)
[Sbud, order] = sort(S.budget)
S = S(order,:)

% Figure 1 - training objective versus evaluation budget
f = figure('Visible','off','Color','w','Position',[100 100 950 530])
plot(Sbud,S.random_train,'-o','LineWidth',2,'MarkerSize',8)
hold on
plot(Sbud,S.coordinate_train,'-s','LineWidth',2,'MarkerSize',8)
plot(Sbud,S.oracle,'--','LineWidth',1.7)
hold off
grid on
xlabel('Objective evaluations (budget)')
ylabel('Mean training surrogate score')
title('Synthetic benchmark, length 3, 50 seeds per method')
legend('Random search','Coordinate search','Exact oracle','Location','southeast')
exportgraphics(f,fullfile(out_dir,'matlab_score_budget.png'),'Resolution',220)
close(f)

% Figure 2 - regret versus prompt length for 36 score calls
S = T(T.budget == 36,:)
[Slen, order] = sort(S.length)
S = S(order,:)
f = figure('Visible','off','Color','w','Position',[100 100 950 530])
plot(Slen,S.random_regret,'-o','LineWidth',2,'MarkerSize',8)
hold on
plot(Slen,S.coordinate_regret,'-s','LineWidth',2,'MarkerSize',8)
hold off
grid on
xticks([2 3 4])
xlabel('Token sequence length')
ylabel('Mean training simple regret')
title('Synthetic benchmark, budget 36, lower is better')
legend('Random search','Coordinate search','Location','northwest')
exportgraphics(f,fullfile(out_dir,'matlab_regret_length.png'),'Resolution',220)
close(f)

% Figure 3 - development and held-out synthetic objectives
S = T(T.length == 3 & T.budget == 36,:)
vals = [S.random_train S.random_holdout; S.coordinate_train S.coordinate_holdout]
f = figure('Visible','off','Color','w','Position',[100 100 950 530])
bar(vals)
grid on
xticklabels({'Random search','Coordinate search'})
ylabel('Mean surrogate score')
title('Train and held-out synthetic targets, length 3, budget 36')
legend('Development','Held-out','Location','southwest')
exportgraphics(f,fullfile(out_dir,'matlab_train_holdout.png'),'Resolution',220)
close(f)

% Figure 4 - paired-seed differences and exploratory bootstrap intervals
S = T(T.length == 3,:)
[Sbud, order] = sort(S.budget)
S = S(order,:)
f = figure('Visible','off','Color','w','Position',[100 100 950 530])
lower_span = S.paired_difference - S.paired_ci_lower
upper_span = S.paired_ci_upper - S.paired_difference
errorbar(Sbud,S.paired_difference,lower_span,upper_span,'-o','LineWidth',2,'MarkerSize',8)
yline(0,'--')
grid on
xlabel('Objective evaluations (budget)')
ylabel('Mean paired difference: coordinate minus random')
title('Synthetic seeds only, exploratory 95% bootstrap intervals')
exportgraphics(f,fullfile(out_dir,'matlab_paired_difference.png'),'Resolution',220)
close(f)
disp('Four synthetic result figures written to tex/figures')
end
