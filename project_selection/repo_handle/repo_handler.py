import os
import logging
from tempfile import mkdtemp
from git import Repo
from shutil import copytree
from options import Options
class RepositoryHandler:
    def __init__(self, repo_full_name, repo_url, repos_dir=None):
        """
        Initialize the repository handler, clone if necessary, and set up paths.

        :param repo_full_name: Full name of the repository (e.g., "user/repo").
        :param repo_url: URL to clone the repository if not found locally.
        :param repos_dir: Directory to search for a local copy of the repository.
        """
        self._repository = None

        # Step 1: Set up the temporary directory
        os.makedirs(Options.TEMP_WORKING_DIR, exist_ok=True)
        self.__temp_dir = mkdtemp(dir=os.path.join(os.getcwd(), Options.TEMP_WORKING_DIR))
        logging.info(f"Created a temp directory: {self.__temp_dir}")

        # Step 2: Define repository path in the temp directory
        self._repository_path = os.path.join(self.__temp_dir, repo_full_name.replace("/", "_"))

        # Step 3: Check if repository exists locally or needs to be cloned
        if not os.path.isdir(self._repository_path):
            if repos_dir:
                # If local repo directory is provided, try copying the repo
                repo_dir = os.path.join(repos_dir, repo_full_name)
                if os.path.isdir(repo_dir):
                    logging.info(f"Copying local repository: {repo_dir}")
                    copytree(repo_dir, self._repository_path, symlinks=True)
                else:
                    logging.error(f"Unable to find local repository path: {repo_dir}")
                    exit(-4)  # Exit with an error if the local repo is missing
            else:
                # Clone from remote if no local copy is found
                logging.info(f"Cloning repository {repo_full_name} from {repo_url}...")
                Repo.clone_from(url=repo_url, to_path=self._repository_path)

        # Step 4: Set up the repository instance
        self._repository = Repo(self._repository_path)

if __name__ == "__main__":
    repo = RepositoryHandler(repo_full_name="", repo_url="")
