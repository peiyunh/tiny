function plot_pr(propose,recall,lendge_name,seting_class,setting_name,dateset_class)
model_num = size(propose,1);
figure1 = figure('PaperSize',[20.98 29.68],'Color',[1 1 1]);
axes1 = axes('Parent',figure1,...
    'LineWidth',2,...
    'FontSize',15,...
    'FontName','Times New Roman',...
    'FontWeight','bold');
box(axes1,'on');
hold on;

LineColor = colormap(hsv(model_num));
for i=1:model_num
    plot(propose{i},recall{i},...
        'MarkerEdgeColor',LineColor(i,:),...
        'MarkerFaceColor',LineColor(i,:),...
        'LineWidth',4,...
        'Color',LineColor(i,:))
    hleg = legend(lendge_name{:},'Location','SouthEast');
    grid on;
    hold on;
end
xlim([0,1]);
ylim([0,1]);
xlabel('Recall');
ylabel('Precision');

savename = sprintf('./plot/figure/%s/wider_pr_cruve_%s_%s.pdf',dateset_class,seting_class,setting_name);
saveTightFigure(gcf,savename);
clear gcf;
hold off;




