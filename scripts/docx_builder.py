"""
ECG DigiAI - Final Project Report Generator (.docx)
Adheres strictly to Project Report Structure.docx (NBA Framework)
Generates complete Microsoft Word document without requiring python-docx.
"""

import os
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image

import scripts.report_data as rd

def escape_xml(text: str) -> str:
    """Escapes special characters for XML."""
    if text is None:
        return ""
    text = str(text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

class DocxReportBuilder:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.doc_body_xml = []
        self.relationships = []  # list of (rId, type, target)
        self.r_id_counter = 5    # rId 1-4 reserved for styles, headers, footers
        self.images_to_pack = {} # rId -> local_path
        self.media_counter = 1
        self.current_section = 1
        
    def _add_p(self, runs_xml: str, style: str = "Normal", align: str = "both", 
               space_before: int = 0, space_after: int = 120, line_spacing: int = 360,
               keep_next: bool = False, page_break_before: bool = False):
        p_pr = [f'<w:pStyle w:val="{style}"/>']
        if align != "both":
            p_pr.append(f'<w:jc w:val="{align}"/>')
        if space_before > 0 or space_after > 0 or line_spacing != 240:
            p_pr.append(f'<w:spacing w:before="{space_before}" w:after="{space_after}" w:line="{line_spacing}" w:lineRule="auto"/>')
        if keep_next:
            p_pr.append('<w:keepNext/>')
        if page_break_before:
            p_pr.append('<w:pageBreakBefore/>')
            
        xml = f'<w:p><w:pPr>{"".join(p_pr)}</w:pPr>{runs_xml}</w:p>'
        self.doc_body_xml.append(xml)

    def add_title(self, text: str, subtitle: str = ""):
        run = f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="32"/><w:color w:val="1A365D"/></w:rPr><w:t>{escape_xml(text)}</w:t></w:r>'
        self._add_p(run, style="Heading1", align="center", space_before=120, space_after=120, keep_next=True)
        if subtitle:
            sub_run = f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:i/><w:sz w:val="24"/><w:color w:val="4A5568"/></w:rPr><w:t>{escape_xml(subtitle)}</w:t></w:r>'
            self._add_p(sub_run, style="Normal", align="center", space_before=0, space_after=240, keep_next=True)

    def add_heading1(self, text: str, page_break_before: bool = True):
        run = f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="32"/><w:color w:val="1A365D"/></w:rPr><w:t>{escape_xml(text.upper())}</w:t></w:r>'
        self._add_p(run, style="Heading1", align="left", space_before=240, space_after=140, line_spacing=360, keep_next=True, page_break_before=page_break_before)

    def add_heading2(self, text: str):
        run = f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="28"/><w:color w:val="2B6CB0"/></w:rPr><w:t>{escape_xml(text)}</w:t></w:r>'
        self._add_p(run, style="Heading2", align="left", space_before=200, space_after=80, line_spacing=360, keep_next=True)

    def add_heading3(self, text: str):
        run = f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="24"/><w:color w:val="2D3748"/></w:rPr><w:t>{escape_xml(text)}</w:t></w:r>'
        self._add_p(run, style="Heading3", align="left", space_before=140, space_after=60, line_spacing=360, keep_next=True)

    def add_paragraph(self, text: str, bold_prefix: str = "", italic: bool = False, align: str = "both", space_after: int = 120):
        runs = []
        if bold_prefix:
            runs.append(f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="24"/></w:rPr><w:t xml:space="preserve">{escape_xml(bold_prefix)} </w:t></w:r>')
        r_pr = ['<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>', '<w:sz w:val="24"/>']
        if italic:
            r_pr.append('<w:i/>')
        runs.append(f'<w:r><w:rPr>{"".join(r_pr)}</w:rPr><w:t xml:space="preserve">{escape_xml(text)}</w:t></w:r>')
        self._add_p("".join(runs), style="Normal", align=align, space_before=0, space_after=space_after, line_spacing=360)

    def add_bullet(self, text: str, bold_prefix: str = ""):
        runs = []
        runs.append(f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr><w:t xml:space="preserve">&#x2022;  </w:t></w:r>')
        if bold_prefix:
            runs.append(f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="24"/></w:rPr><w:t xml:space="preserve">{escape_xml(bold_prefix)} </w:t></w:r>')
        runs.append(f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr><w:t xml:space="preserve">{escape_xml(text)}</w:t></w:r>')
        
        p_pr = [
            '<w:pStyle w:val="Normal"/>',
            '<w:jc w:val="both"/>',
            '<w:ind w:left="480" w:hanging="240"/>',
            '<w:spacing w:before="0" w:after="80" w:line="360" w:lineRule="auto"/>'
        ]
        xml = f'<w:p><w:pPr>{"".join(p_pr)}</w:pPr>{"".join(runs)}</w:p>'
        self.doc_body_xml.append(xml)

    def add_callout(self, text: str, title: str = "ACADEMIC DECISION-SUPPORT NOTICE"):
        p_pr = [
            '<w:pStyle w:val="Normal"/>',
            '<w:jc w:val="both"/>',
            '<w:pBdr><w:left w:val="single" w:sz="24" w:space="12" w:color="2B6CB0"/></w:pBdr>',
            '<w:shd w:val="clear" w:color="auto" w:fill="EDF2F7"/>',
            '<w:ind w:left="360" w:right="360"/>',
            '<w:spacing w:before="120" w:after="160" w:line="300" w:lineRule="auto"/>'
        ]
        run_title = f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="20"/><w:color w:val="1A365D"/></w:rPr><w:t xml:space="preserve">{escape_xml(title.upper())}: </w:t></w:r>'
        run_text = f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:i/><w:sz w:val="20"/><w:color w:val="2D3748"/></w:rPr><w:t>{escape_xml(text)}</w:t></w:r>'
        xml = f'<w:p><w:pPr>{"".join(p_pr)}</w:pPr>{run_title}{run_text}</w:p>'
        self.doc_body_xml.append(xml)

    def add_page_break(self):
        self.doc_body_xml.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def add_section_break_to_main(self):
        """Converts preceding preliminary section to Roman numerals and sets up Arabic numbering."""
        sect_xml = (
            '<w:pPr>'
            '<w:sectPr>'
            '<w:headerReference w:type="default" r:id="rIdHeader1"/>'
            '<w:footerReference w:type="default" r:id="rIdFooter1"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="2160" w:header="720" w:footer="720" w:gutter="0"/>'
            '<w:pgNumType w:fmt="lowerRoman" w:start="1"/>'
            '<w:cols w:space="720"/>'
            '<w:docGrid w:linePitch="360"/>'
            '</w:sectPr>'
            '</w:pPr>'
        )
        self.doc_body_xml.append(f'<w:p>{sect_xml}</w:p>')
        self.current_section = 2

    def add_table(self, headers: list, rows: list, col_widths: list, align_cols: list = None, header_bg: str = "1A365D"):
        """
        Builds a professionally styled academic table.
        col_widths: list of int in dxa (total should equal ~8306)
        """
        if align_cols is None:
            align_cols = ["left"] * len(headers)

        table_xml = []
        table_xml.append('<w:tbl>')
        table_xml.append(
            '<w:tblPr>'
            '<w:tblW w:w="8306" w:type="dxa"/>'
            '<w:jc w:val="center"/>'
            '<w:tblBorders>'
            f'<w:top w:val="single" w:sz="12" w:space="0" w:color="{header_bg}"/>'
            '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="718096"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
            '<w:insideV w:val="none"/>'
            '<w:left w:val="none"/>'
            '<w:right w:val="none"/>'
            '</w:tblBorders>'
            '<w:tblCellMar>'
            '<w:top w:w="120" w:type="dxa"/>'
            '<w:bottom w:w="120" w:type="dxa"/>'
            '<w:left w:w="140" w:type="dxa"/>'
            '<w:right w:w="140" w:type="dxa"/>'
            '</w:tblCellMar>'
            '</w:tblPr>'
        )
        
        # Grid
        table_xml.append('<w:tblGrid>')
        for w in col_widths:
            table_xml.append(f'<w:gridCol w:w="{w}"/>')
        table_xml.append('</w:tblGrid>')

        # Header Row
        table_xml.append('<w:tr><w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>')
        for i, h in enumerate(headers):
            w = col_widths[i]
            align = align_cols[i]
            table_xml.append(
                f'<w:tc><w:tcPr>'
                f'<w:tcW w:w="{w}" w:type="dxa"/>'
                f'<w:shd w:val="clear" w:color="auto" w:fill="{header_bg}"/>'
                f'<w:vAlign w:val="center"/>'
                f'</w:tcPr>'
                f'<w:p><w:pPr><w:jc w:val="{align}"/><w:spacing w:before="60" w:after="60" w:line="240" w:lineRule="auto"/></w:pPr>'
                f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="21"/><w:color w:val="FFFFFF"/></w:rPr>'
                f'<w:t>{escape_xml(h)}</w:t></w:r></w:p></w:tc>'
            )
        table_xml.append('</w:tr>')

        # Data Rows
        for row_idx, r in enumerate(rows):
            bg_color = "F7FAFC" if row_idx % 2 == 1 else "FFFFFF"
            table_xml.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>')
            for i, val in enumerate(r):
                w = col_widths[i]
                align = align_cols[i]
                table_xml.append(
                    f'<w:tc><w:tcPr>'
                    f'<w:tcW w:w="{w}" w:type="dxa"/>'
                    f'<w:shd w:val="clear" w:color="auto" w:fill="{bg_color}"/>'
                    f'<w:vAlign w:val="center"/>'
                    f'</w:tcPr>'
                    f'<w:p><w:pPr><w:jc w:val="{align}"/><w:spacing w:before="60" w:after="60" w:line="240" w:lineRule="auto"/></w:pPr>'
                    f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="21"/><w:color w:val="2D3748"/></w:rPr>'
                    f'<w:t>{escape_xml(str(val))}</w:t></w:r></w:p></w:tc>'
                )
            table_xml.append('</w:tr>')

        table_xml.append('</w:tbl>')
        table_xml.append('<w:p><w:pPr><w:spacing w:before="0" w:after="140" w:line="240" w:lineRule="auto"/></w:pPr></w:p>')
        self.doc_body_xml.append("".join(table_xml))

    def add_figure(self, img_path: str, fig_title: str, caption_text: str):
        """Adds an embedded figure image with formal caption."""
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Figure file not found: {img_path}")

        r_id = f"rIdFig{self.media_counter}"
        self.media_counter += 1
        self.images_to_pack[r_id] = img_path
        self.relationships.append((r_id, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"media/image{self.media_counter-1}.png"))

        im = Image.open(img_path)
        w, h = im.size
        aspect = h / w

        # Printable width: 5.5 inches = 5029200 EMUs
        max_w = int(5.5 * 914400)
        max_h = int(6.2 * 914400)
        cx = max_w
        cy = int(cx * aspect)
        if cy > max_h:
            cy = max_h
            cx = int(cy / aspect)

        drawing_xml = (
            '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="140" w:after="80"/></w:pPr>'
            '<w:r><w:drawing>'
            '<wp:inline distT="0" distB="0" distL="0" distR="0" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{self.media_counter}" name="{escape_xml(fig_title)}"/>'
            '<wp:cNvGraphicFramePr>'
            '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
            '</wp:cNvGraphicFramePr>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:nvPicPr>'
            f'<pic:cNvPr id="{self.media_counter}" name="{escape_xml(fig_title)}"/>'
            '<pic:cNvPicPr/>'
            '</pic:nvPicPr>'
            '<pic:blipFill>'
            f'<a:blip r:embed="{r_id}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
            '<a:stretch><a:fillRect/></a:stretch>'
            '</pic:blipFill>'
            '<pic:spPr>'
            f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '</pic:spPr>'
            '</pic:pic>'
            '</a:graphicData>'
            '</a:graphic>'
            '</wp:inline>'
            '</w:drawing></w:r></w:p>'
        )
        self.doc_body_xml.append(drawing_xml)

        # Caption
        caption_run = (
            f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="20"/><w:color w:val="1A365D"/></w:rPr>'
            f'<w:t>{escape_xml(fig_title)}: </w:t></w:r>'
            f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:i/><w:sz w:val="20"/><w:color w:val="4A5568"/></w:rPr>'
            f'<w:t>{escape_xml(caption_text)}</w:t></w:r>'
        )
        self._add_p(caption_run, style="Caption", align="center", space_before=40, space_after=180, line_spacing=240)

    def generate(self):
        """Assembles and writes the complete .docx zip package."""
        print(f"[*] Assembling Word OpenXML document: {self.output_path}...")

        # Section 2 terminal properties
        sect2_xml = (
            '<w:sectPr>'
            '<w:headerReference w:type="default" r:id="rIdHeader1"/>'
            '<w:footerReference w:type="default" r:id="rIdFooter2"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="2160" w:header="720" w:footer="720" w:gutter="0"/>'
            '<w:pgNumType w:fmt="decimal" w:start="1"/>'
            '<w:cols w:space="720"/>'
            '<w:docGrid w:linePitch="360"/>'
            '</w:sectPr>'
        )
        full_document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">\n'
            '<w:body>\n'
            + "\n".join(self.doc_body_xml) + "\n"
            + sect2_xml + "\n"
            '</w:body>\n'
            '</w:document>'
        )

        # Styles XML
        styles_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
            '  <w:docDefaults>\n'
            '    <w:rPrDefault>\n'
            '      <w:rPr>\n'
            '        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>\n'
            '        <w:sz w:val="24"/>\n'
            '        <w:szCs w:val="24"/>\n'
            '        <w:color w:val="000000"/>\n'
            '      </w:rPr>\n'
            '    </w:rPrDefault>\n'
            '    <w:pPrDefault>\n'
            '      <w:pPr>\n'
            '        <w:spacing w:line="360" w:lineRule="auto" w:after="120"/>\n'
            '        <w:jc w:val="both"/>\n'
            '      </w:pPr>\n'
            '    </w:pPrDefault>\n'
            '  </w:docDefaults>\n'
            '  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">\n'
            '    <w:name w:val="Normal"/>\n'
            '    <w:qFormat/>\n'
            '  </w:style>\n'
            '  <w:style w:type="paragraph" w:styleId="Heading1">\n'
            '    <w:name w:val="heading 1"/>\n'
            '    <w:basedOn w:val="Normal"/>\n'
            '    <w:next w:val="Normal"/>\n'
            '    <w:qFormat/>\n'
            '    <w:pPr>\n'
            '      <w:spacing w:before="240" w:after="120" w:line="360" w:lineRule="auto"/>\n'
            '      <w:jc w:val="left"/>\n'
            '      <w:keepNext/>\n'
            '    </w:pPr>\n'
            '    <w:rPr>\n'
            '      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>\n'
            '      <w:b/>\n'
            '      <w:sz w:val="32"/>\n'
            '      <w:color w:val="1A365D"/>\n'
            '    </w:rPr>\n'
            '  </w:style>\n'
            '  <w:style w:type="paragraph" w:styleId="Heading2">\n'
            '    <w:name w:val="heading 2"/>\n'
            '    <w:basedOn w:val="Normal"/>\n'
            '    <w:next w:val="Normal"/>\n'
            '    <w:qFormat/>\n'
            '    <w:pPr>\n'
            '      <w:spacing w:before="200" w:after="80" w:line="360" w:lineRule="auto"/>\n'
            '      <w:jc w:val="left"/>\n'
            '      <w:keepNext/>\n'
            '    </w:pPr>\n'
            '    <w:rPr>\n'
            '      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>\n'
            '      <w:b/>\n'
            '      <w:sz w:val="28"/>\n'
            '      <w:color w:val="2B6CB0"/>\n'
            '    </w:rPr>\n'
            '  </w:style>\n'
            '  <w:style w:type="paragraph" w:styleId="Heading3">\n'
            '    <w:name w:val="heading 3"/>\n'
            '    <w:basedOn w:val="Normal"/>\n'
            '    <w:next w:val="Normal"/>\n'
            '    <w:qFormat/>\n'
            '    <w:pPr>\n'
            '      <w:spacing w:before="140" w:after="60" w:line="360" w:lineRule="auto"/>\n'
            '      <w:jc w:val="left"/>\n'
            '      <w:keepNext/>\n'
            '    </w:pPr>\n'
            '    <w:rPr>\n'
            '      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>\n'
            '      <w:b/>\n'
            '      <w:sz w:val="24"/>\n'
            '      <w:color w:val="2D3748"/>\n'
            '    </w:rPr>\n'
            '  </w:style>\n'
            '  <w:style w:type="paragraph" w:styleId="Caption">\n'
            '    <w:name w:val="caption"/>\n'
            '    <w:basedOn w:val="Normal"/>\n'
            '    <w:qFormat/>\n'
            '    <w:pPr>\n'
            '      <w:spacing w:before="40" w:after="160" w:line="240" w:lineRule="auto"/>\n'
            '      <w:jc w:val="center"/>\n'
            '    </w:pPr>\n'
            '    <w:rPr>\n'
            '      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>\n'
            '      <w:i/>\n'
            '      <w:sz w:val="20"/>\n'
            '      <w:color w:val="4A5568"/>\n'
            '    </w:rPr>\n'
            '  </w:style>\n'
            '</w:styles>'
        )

        # Header 1 XML (Running Header)
        header1_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
            '  <w:p>\n'
            '    <w:pPr>\n'
            '      <w:pBdr><w:bottom w:val="single" w:sz="6" w:space="4" w:color="CBD5E0"/></w:pBdr>\n'
            '      <w:jc w:val="right"/>\n'
            '      <w:spacing w:before="0" w:after="60" w:line="240" w:lineRule="auto"/>\n'
            '    </w:pPr>\n'
            '    <w:r>\n'
            '      <w:rPr>\n'
            '        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>\n'
            '        <w:sz w:val="18"/>\n'
            '        <w:color w:val="718096"/>\n'
            '      </w:rPr>\n'
            '      <w:t>JAYPEE INSTITUTE OF INFORMATION TECHNOLOGY, NOIDA | B.TECH MAJOR PROJECT: ECG DIGIAI</w:t>\n'
            '    </w:r>\n'
            '  </w:p>\n'
            '</w:hdr>'
        )

        # Footer 1 XML (Preliminary Roman Numbering)
        footer1_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
            '  <w:p>\n'
            '    <w:pPr>\n'
            '      <w:jc w:val="center"/>\n'
            '      <w:spacing w:before="60" w:after="0" w:line="240" w:lineRule="auto"/>\n'
            '    </w:pPr>\n'
            '    <w:r>\n'
            '      <w:rPr>\n'
            '        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>\n'
            '        <w:sz w:val="20"/>\n'
            '        <w:color w:val="4A5568"/>\n'
            '      </w:rPr>\n'
            '      <w:fldSimple w:instr="PAGE"/>\n'
            '    </w:r>\n'
            '  </w:p>\n'
            '</w:ftr>'
        )

        # Footer 2 XML (Main Chapters Arabic Numbering)
        footer2_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
            '  <w:p>\n'
            '    <w:pPr>\n'
            '      <w:pBdr><w:top w:val="single" w:sz="6" w:space="4" w:color="CBD5E0"/></w:pBdr>\n'
            '      <w:jc w:val="center"/>\n'
            '      <w:spacing w:before="60" w:after="0" w:line="240" w:lineRule="auto"/>\n'
            '    </w:pPr>\n'
            '    <w:r>\n'
            '      <w:rPr>\n'
            '        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>\n'
            '        <w:sz w:val="20"/>\n'
            '        <w:color w:val="4A5568"/>\n'
            '      </w:rPr>\n'
            '      <w:t xml:space="preserve">Page </w:t>\n'
            '      <w:fldSimple w:instr="PAGE"/>\n'
            '      <w:t xml:space="preserve"> of </w:t>\n'
            '      <w:fldSimple w:instr="NUMPAGES"/>\n'
            '    </w:r>\n'
            '  </w:p>\n'
            '</w:ftr>'
        )

        # Content Types XML
        content_types_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '  <Default Extension="xml" ContentType="application/xml"/>\n'
            '  <Default Extension="png" ContentType="image/png"/>\n'
            '  <Default Extension="jpeg" ContentType="image/jpeg"/>\n'
            '  <Default Extension="jpg" ContentType="image/jpeg"/>\n'
            '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
            '  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>\n'
            '  <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>\n'
            '  <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>\n'
            '  <Override PartName="/word/footer2.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>\n'
            '</Types>'
        )

        # Root Relationships XML (_rels/.rels)
        root_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>\n'
            '</Relationships>'
        )

        # Document Relationships XML (word/_rels/document.xml.rels)
        doc_rels = [
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
            '<Relationship Id="rIdHeader1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>',
            '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>',
            '<Relationship Id="rIdFooter2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer2.xml"/>'
        ]
        for r_id, r_type, r_target in self.relationships:
            doc_rels.append(f'<Relationship Id="{r_id}" Type="{r_type}" Target="{r_target}"/>')

        doc_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            + "\n".join(doc_rels) + "\n"
            '</Relationships>'
        )

        # Write to Zip
        with zipfile.ZipFile(self.output_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types_xml)
            z.writestr("_rels/.rels", root_rels_xml)
            z.writestr("word/_rels/document.xml.rels", doc_rels_xml)
            z.writestr("word/styles.xml", styles_xml)
            z.writestr("word/header1.xml", header1_xml)
            z.writestr("word/footer1.xml", footer1_xml)
            z.writestr("word/footer2.xml", footer2_xml)
            z.writestr("word/document.xml", full_document_xml)

            # Media files
            for r_id, local_file in self.images_to_pack.items():
                for rel_id, rel_type, rel_target in self.relationships:
                    if rel_id == r_id:
                        arcname = f"word/{rel_target}"
                        with open(local_file, "rb") as f_img:
                            z.writestr(arcname, f_img.read())
                        break

        print(f"[+] Successfully generated Word document: {self.output_path} ({os.path.getsize(self.output_path):,} bytes)")

print("[+] DocxReportBuilder class loaded successfully.")
