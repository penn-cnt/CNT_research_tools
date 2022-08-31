function [out_values,car_labels] = common_average_reference(values,include,labels)

out_values = values - repmat(nanmean(values(:,include),2),1,size(values,2));

car_labels = labels;
for i = 1:length(labels)
    car_labels{i} = [labels{i},'-CAR'];
end

end