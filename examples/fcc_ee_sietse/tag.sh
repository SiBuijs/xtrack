set -e

VAR=$(cat VERSION)

echo "Tagging version $VAR"
git tag -a "$VAR" -m "Version $VAR"
git push --tags