# 翻译skill使用方法

1. 将exclude_docs.json和SKILL.md下载到本地同一目录存放（如`D:\skills\`）。从SKILL.md所在目录读取排除配置`exclude_docs.json`，与待翻译仓库的位置无关；该目录下无配置文件时视为无排除项，整个仓库进行翻译。

2. 修改exclude_docs.json文件中需要排除翻译的信息：

   - `exclude_docs.json`中的`excludeDocsPaths`支持自定义配置，可根据仓库实际情况调整，用于过滤无需翻译的内容。文件内配置仅为示例，请结合各仓库自身情况修改。各仓库需要排除的目录参考[白名单](https://gitcode.com/cann/docs/blob/master/skills/translation_skill/repo_zh_en_ci_check_whitelist.md)。
   - `repoName`为仓库名，skill会将用户输入的本地仓库路径与`repoName`进行模糊匹配（路径中包含repoName关键字即匹配成功）；匹配成功则按`excludeDocsPaths`排除翻译，未匹配则整个仓库进行翻译。

3. 在opencode中输入类似如下信息，启动skill翻译任务：

   根据\<skill_dir>\SKILL.md skill，处理xxx仓或yyy.md文件。其中，\<skill_dir>为exclude_docs.json和SKILL.md所在目录，xxx为用户clone到本地的组件仓库，yyy为仓中的md文档。

   skill启动后会交互式提示"请输入需要翻译的本地仓库路径"，此时输入clone到本地的仓库路径（如`D:\WorkSpace\metadef_trunk`）即可。

   说明：上述翻译skill会将翻译后的英文文档提交到远程仓库，如果不希望提交，可以在上面的描述后面补充：翻译完成后，无需提交到远程仓库。
