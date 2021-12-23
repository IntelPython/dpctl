#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2021 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json


def create_extlinks():
    """Reads a JSON file to create a dictionary of urls in the format supported
    by the sphinx.ect.extlinks extension.

    Returns:
        dict: A dictionary that is understood by the extlinks Sphinx extension.

    """
    extlinks = {}

    with open("docfiles/urls.json") as urls_json:
        urls = json.load(urls_json)
        for url in urls:
            url_value = urls[url]
            extlinks[url] = (url_value + "%s", None)

    return extlinks
