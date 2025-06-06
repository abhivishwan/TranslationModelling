function [b] = add_externalXlabelsTop(wg, hg, wb, hb, delta2, str)


% delta2 = 0.06;
for i = 1:length(str)
    b(i) = annotation('textbox', [sum(wg(1:i))+sum(wb(1:i-1)) sum(hg(1:end-1))+sum(hb)+delta2 wb(i) hg(end)-delta2], 'string', str{i});
end
set(b, 'fontsize', 12, 'linestyle', 'none', ...
    'horizontalalignment', 'center', ...
    'verticalalignment', 'middle')
