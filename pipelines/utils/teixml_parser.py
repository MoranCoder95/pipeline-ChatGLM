# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 deepset GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lxml import etree

from lxml import etree
import re

def parse_tei_xml(xml_content):
    nsmap = {'tei': 'http://www.tei-c.org/ns/1.0'}
    root = etree.fromstring(xml_content)

    #  Extract the title element and its text
    title_element = root.find('.//tei:title', namespaces=nsmap)
    title = title_element.text if title_element is not None else None

    #  Extract the analytic element which contains author information
    analytic_element = root.find('.//tei:analytic', namespaces=nsmap)
    authors_elements = analytic_element.findall('.//tei:author', namespaces=nsmap) if analytic_element is not None else []
    author_information_list = []

    for author_element in authors_elements:
        forename = author_element.find('tei:persName/tei:forename', namespaces=nsmap)
        surname = author_element.find('tei:persName/tei:surname', namespaces=nsmap)

        department = author_element.find('tei:affiliation/tei:orgName[@type="department"]', namespaces=nsmap)
        institution = author_element.find('tei:affiliation/tei:orgName[@type="institution"]', namespaces=nsmap)

        address_element = author_element.find('tei:affiliation/tei:address', namespaces=nsmap)
        address = ', '.join(address_element.itertext()).replace('\n', '').replace('\t', '') if address_element is not None else ''

        author_name = ' '.join([name.text for name in [forename, surname] if name is not None])

        if not author_name:
            continue

        department = department.text if department is not None else ''
        institution = institution.text if institution is not None else ''

        author_information = ', '.join(filter(None, [author_name.strip(), department.strip(), institution.strip(), address.strip()]))
        author_information_list.append(author_information)

    # Extract the keywords elements and their text
    keywords_elements = root.findall('.//tei:keywords/tei:term', namespaces=nsmap)
    keywords = [keyword.text for keyword in keywords_elements]

    # Extract the abstract element and its text
    abstract_element = root.find('.//tei:abstract', namespaces=nsmap)
    abstract = ' '.join(abstract_element.itertext()).replace('\n', '').replace('\t', '') if abstract_element is not None else ''

    # Extract the body content
    body_element = root.find('.//tei:body', namespaces=nsmap)
    body= ' '.join(body_element.itertext()).replace('\t', '') if body_element is not None else ''
    clean_body = re.sub('\n+', '\n', body)

    # Combine
    result_text = f'Title: {title}\n'
    result_text += '\n'.join(author_information_list) + '\n'
    result_text += f'Keywords: {", ".join(keywords)}\n'
    result_text += f'Abstract: {abstract}\n'
    result_text += f'body: {body}\n'

    return result_text



