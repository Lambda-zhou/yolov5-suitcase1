import pkg_resources

installed_packages = sorted([d for d in pkg_resources.working_set])
with open('installed_packages.txt', 'w') as file:
    for package in installed_packages:
        file.write(package.project_name + '\n')