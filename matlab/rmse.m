function varargout= rmse(varargin)

if nargin<2
    error('�������������ڵ���2.');
elseif nargin==2
    varargout{1}=h(varargin{1},varargin{2});
end
end

function r=h(f,g)
%�������ܣ�
%      ����r=h(f,g)�������ͼ��ľ��������r
%���������
%      f----��׼ͼ��
%      g----�ںϺ��ͼ��
%-------------------------------------%

f=double(f);
g=double(g);
[m,n]=size(f);

temp=[];
for i=1:m
    for j=1:n
        temp(i,j)=(f(i,j)-g(i,j))^2;
    end
end

r=sqrt(sum(sum(temp))/(m*n));

end
