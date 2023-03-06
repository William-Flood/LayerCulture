class Field:
    def __int__(self, reshape_list, project_list, einsum_list):
        self._reshape_list = reshape_list
        self._project_list = project_list
        self._einsum_list = einsum_list

    def get_reshape_list(self):
        return self._reshape_list

    def get_project_list(self):
        return self._project_list

    def get_einsum_list(self):
        return self._einsum_list
