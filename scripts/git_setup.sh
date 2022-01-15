git config --global user.name $1
git config --global user.email $4
git config -l

git config --global credential.helper cache


cd
mkdir -p repos
cd repos

# Clone repo, expect git user name and password token
printf "$1\n$2\n" | git clone $3

